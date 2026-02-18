package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"

	"github.com/gin-gonic/gin"
)

// OpenAIGatewayHandler handles OpenAI API gateway requests
type OpenAIGatewayHandler struct {
	gatewayService          *service.OpenAIGatewayService
	billingCacheService     *service.BillingCacheService
	apiKeyService           *service.APIKeyService
	errorPassthroughService *service.ErrorPassthroughService
	concurrencyHelper       *ConcurrencyHelper
	maxAccountSwitches      int
}

// NewOpenAIGatewayHandler creates a new OpenAIGatewayHandler
func NewOpenAIGatewayHandler(
	gatewayService *service.OpenAIGatewayService,
	concurrencyService *service.ConcurrencyService,
	billingCacheService *service.BillingCacheService,
	apiKeyService *service.APIKeyService,
	errorPassthroughService *service.ErrorPassthroughService,
	cfg *config.Config,
) *OpenAIGatewayHandler {
	pingInterval := time.Duration(0)
	maxAccountSwitches := 3
	if cfg != nil {
		pingInterval = time.Duration(cfg.Concurrency.PingInterval) * time.Second
		if cfg.Gateway.MaxAccountSwitches > 0 {
			maxAccountSwitches = cfg.Gateway.MaxAccountSwitches
		}
	}
	return &OpenAIGatewayHandler{
		gatewayService:          gatewayService,
		billingCacheService:     billingCacheService,
		apiKeyService:           apiKeyService,
		errorPassthroughService: errorPassthroughService,
		concurrencyHelper:       NewConcurrencyHelper(concurrencyService, SSEPingFormatComment, pingInterval),
		maxAccountSwitches:      maxAccountSwitches,
	}
}

// Responses handles OpenAI Responses API endpoint
// POST /openai/v1/responses
func (h *OpenAIGatewayHandler) Responses(c *gin.Context) {
	// Get apiKey and user from context (set by ApiKeyAuth middleware)
	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}

	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusInternalServerError, "api_error", "User context not found")
		return
	}

	// Read request body
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		if maxErr, ok := extractMaxBytesError(err); ok {
			h.errorResponse(c, http.StatusRequestEntityTooLarge, "invalid_request_error", buildBodyTooLargeMessage(maxErr.Limit))
			return
		}
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}

	if len(body) == 0 {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
		return
	}

	setOpsRequestContext(c, "", false, body)

	// Parse request body to map for potential modification
	var reqBody map[string]any
	if err := json.Unmarshal(body, &reqBody); err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse request body")
		return
	}

	// Extract model and stream
	reqModel, _ := reqBody["model"].(string)
	reqStream, _ := reqBody["stream"].(bool)

	// 验证 model 必填
	if reqModel == "" {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}

	userAgent := c.GetHeader("User-Agent")
	if !openai.IsCodexCLIRequest(userAgent) {
		existingInstructions, _ := reqBody["instructions"].(string)
		if strings.TrimSpace(existingInstructions) == "" {
			if instructions := strings.TrimSpace(service.GetOpenCodeInstructions()); instructions != "" {
				reqBody["instructions"] = instructions
				// Re-serialize body
				body, err = json.Marshal(reqBody)
				if err != nil {
					h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to process request")
					return
				}
			}
		}
	}

	setOpsRequestContext(c, reqModel, reqStream, body)

	// 提前校验 function_call_output 是否具备可关联上下文，避免上游 400。
	// 要求 previous_response_id，或 input 内存在带 call_id 的 tool_call/function_call，
	// 或带 id 且与 call_id 匹配的 item_reference。
	if service.HasFunctionCallOutput(reqBody) {
		previousResponseID, _ := reqBody["previous_response_id"].(string)
		if strings.TrimSpace(previousResponseID) == "" && !service.HasToolCallContext(reqBody) {
			if service.HasFunctionCallOutputMissingCallID(reqBody) {
				log.Printf("[OpenAI Handler] function_call_output 缺少 call_id: model=%s", reqModel)
				h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "function_call_output requires call_id or previous_response_id; if relying on history, ensure store=true and reuse previous_response_id")
				return
			}
			callIDs := service.FunctionCallOutputCallIDs(reqBody)
			if !service.HasItemReferenceForCallIDs(reqBody, callIDs) {
				log.Printf("[OpenAI Handler] function_call_output 缺少匹配的 item_reference: model=%s", reqModel)
				h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "function_call_output requires item_reference ids matching each call_id, or previous_response_id/tool_call context; if relying on history, ensure store=true and reuse previous_response_id")
				return
			}
		}
	}

	// Track if we've started streaming (for error handling)
	streamStarted := false

	// 绑定错误透传服务，允许 service 层在非 failover 错误场景复用规则。
	if h.errorPassthroughService != nil {
		service.BindErrorPassthroughService(c, h.errorPassthroughService)
	}

	// Get subscription info (may be nil)
	subscription, _ := middleware2.GetSubscriptionFromContext(c)

	// 0. Check if wait queue is full
	maxWait := service.CalculateMaxWait(subject.Concurrency)
	canWait, err := h.concurrencyHelper.IncrementWaitCount(c.Request.Context(), subject.UserID, maxWait)
	waitCounted := false
	if err != nil {
		log.Printf("Increment wait count failed: %v", err)
		// On error, allow request to proceed
	} else if !canWait {
		h.errorResponse(c, http.StatusTooManyRequests, "rate_limit_error", "Too many pending requests, please retry later")
		return
	}
	if err == nil && canWait {
		waitCounted = true
	}
	defer func() {
		if waitCounted {
			h.concurrencyHelper.DecrementWaitCount(c.Request.Context(), subject.UserID)
		}
	}()

	// 1. First acquire user concurrency slot
	userReleaseFunc, err := h.concurrencyHelper.AcquireUserSlotWithWait(c, subject.UserID, subject.Concurrency, reqStream, &streamStarted)
	if err != nil {
		log.Printf("User concurrency acquire failed: %v", err)
		h.handleConcurrencyError(c, err, "user", streamStarted)
		return
	}
	// User slot acquired: no longer waiting.
	if waitCounted {
		h.concurrencyHelper.DecrementWaitCount(c.Request.Context(), subject.UserID)
		waitCounted = false
	}
	// 确保请求取消时也会释放槽位，避免长连接被动中断造成泄漏
	userReleaseFunc = wrapReleaseOnDone(c.Request.Context(), userReleaseFunc)
	if userReleaseFunc != nil {
		defer userReleaseFunc()
	}

	// 2. Re-check billing eligibility after wait
	if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription); err != nil {
		log.Printf("Billing eligibility check failed after wait: %v", err)
		status, code, message := billingErrorDetails(err)
		h.handleStreamingAwareError(c, status, code, message, streamStarted)
		return
	}

	// Generate session hash (header first; fallback to prompt_cache_key)
	sessionHash := h.gatewayService.GenerateSessionHash(c, reqBody)

	maxAccountSwitches := h.maxAccountSwitches
	switchCount := 0
	failedAccountIDs := make(map[int64]struct{})
	var lastFailoverErr *service.UpstreamFailoverError

	for {
		// Select account supporting the requested model
		log.Printf("[OpenAI Handler] Selecting account: groupID=%v model=%s", apiKey.GroupID, reqModel)
		selection, err := h.gatewayService.SelectAccountWithLoadAwareness(c.Request.Context(), apiKey.GroupID, sessionHash, reqModel, failedAccountIDs)
		if err != nil {
			log.Printf("[OpenAI Handler] SelectAccount failed: %v", err)
			if len(failedAccountIDs) == 0 {
				h.handleStreamingAwareError(c, http.StatusServiceUnavailable, "api_error", "No available accounts: "+err.Error(), streamStarted)
				return
			}
			if lastFailoverErr != nil {
				h.handleFailoverExhausted(c, lastFailoverErr, streamStarted)
			} else {
				h.handleFailoverExhaustedSimple(c, 502, streamStarted)
			}
			return
		}
		account := selection.Account
		log.Printf("[OpenAI Handler] Selected account: id=%d name=%s", account.ID, account.Name)
		setOpsSelectedAccount(c, account.ID)

		// 3. Acquire account concurrency slot
		accountReleaseFunc := selection.ReleaseFunc
		if !selection.Acquired {
			if selection.WaitPlan == nil {
				h.handleStreamingAwareError(c, http.StatusServiceUnavailable, "api_error", "No available accounts", streamStarted)
				return
			}
			accountWaitCounted := false
			canWait, err := h.concurrencyHelper.IncrementAccountWaitCount(c.Request.Context(), account.ID, selection.WaitPlan.MaxWaiting)
			if err != nil {
				log.Printf("Increment account wait count failed: %v", err)
			} else if !canWait {
				log.Printf("Account wait queue full: account=%d", account.ID)
				h.handleStreamingAwareError(c, http.StatusTooManyRequests, "rate_limit_error", "Too many pending requests, please retry later", streamStarted)
				return
			}
			if err == nil && canWait {
				accountWaitCounted = true
			}
			defer func() {
				if accountWaitCounted {
					h.concurrencyHelper.DecrementAccountWaitCount(c.Request.Context(), account.ID)
				}
			}()

			accountReleaseFunc, err = h.concurrencyHelper.AcquireAccountSlotWithWaitTimeout(
				c,
				account.ID,
				selection.WaitPlan.MaxConcurrency,
				selection.WaitPlan.Timeout,
				reqStream,
				&streamStarted,
			)
			if err != nil {
				log.Printf("Account concurrency acquire failed: %v", err)
				h.handleConcurrencyError(c, err, "account", streamStarted)
				return
			}
			if accountWaitCounted {
				h.concurrencyHelper.DecrementAccountWaitCount(c.Request.Context(), account.ID)
				accountWaitCounted = false
			}
			if err := h.gatewayService.BindStickySession(c.Request.Context(), apiKey.GroupID, sessionHash, account.ID); err != nil {
				log.Printf("Bind sticky session failed: %v", err)
			}
		}
		// 账号槽位/等待计数需要在超时或断开时安全回收
		accountReleaseFunc = wrapReleaseOnDone(c.Request.Context(), accountReleaseFunc)

		// Forward request
		result, err := h.gatewayService.Forward(c.Request.Context(), c, account, body)
		if accountReleaseFunc != nil {
			accountReleaseFunc()
		}
		if err != nil {
			var failoverErr *service.UpstreamFailoverError
			if errors.As(err, &failoverErr) {
				failedAccountIDs[account.ID] = struct{}{}
				lastFailoverErr = failoverErr
				if switchCount >= maxAccountSwitches {
					h.handleFailoverExhausted(c, failoverErr, streamStarted)
					return
				}
				switchCount++
				log.Printf("Account %d: upstream error %d, switching account %d/%d", account.ID, failoverErr.StatusCode, switchCount, maxAccountSwitches)
				continue
			}
			// Error response already handled in Forward, just log
			log.Printf("Account %d: Forward request failed: %v", account.ID, err)
			return
		}

		// 捕获请求信息（用于异步记录，避免在 goroutine 中访问 gin.Context）
		userAgent := c.GetHeader("User-Agent")
		clientIP := ip.GetClientIP(c)

		// Async record usage
		go func(result *service.OpenAIForwardResult, usedAccount *service.Account, ua, ip string) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := h.gatewayService.RecordUsage(ctx, &service.OpenAIRecordUsageInput{
				Result:        result,
				APIKey:        apiKey,
				User:          apiKey.User,
				Account:       usedAccount,
				Subscription:  subscription,
				UserAgent:     ua,
				IPAddress:     ip,
				APIKeyService: h.apiKeyService,
			}); err != nil {
				log.Printf("Record usage failed: %v", err)
			}
		}(result, account, userAgent, clientIP)
		return
	}
}

// ChatCompletions handles OpenAI Chat Completions compatibility endpoint.
// POST /v1/chat/completions
func (h *OpenAIGatewayHandler) ChatCompletions(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		if maxErr, ok := extractMaxBytesError(err); ok {
			log.Printf("[OpenAI ChatCompat] request body too large: path=%s limit=%d ua=%q", c.Request.URL.Path, maxErr.Limit, c.GetHeader("User-Agent"))
			h.errorResponse(c, http.StatusRequestEntityTooLarge, "invalid_request_error", buildBodyTooLargeMessage(maxErr.Limit))
			return
		}
		log.Printf("[OpenAI ChatCompat] read request body failed: path=%s err=%v ua=%q", c.Request.URL.Path, err, c.GetHeader("User-Agent"))
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}
	if len(body) == 0 {
		log.Printf("[OpenAI ChatCompat] empty request body: path=%s content_type=%q ua=%q", c.Request.URL.Path, c.GetHeader("Content-Type"), c.GetHeader("User-Agent"))
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
		return
	}

	var reqBody map[string]any
	if err := json.Unmarshal(body, &reqBody); err != nil {
		log.Printf("[OpenAI ChatCompat] parse request body failed: path=%s err=%v content_type=%q ua=%q", c.Request.URL.Path, err, c.GetHeader("Content-Type"), c.GetHeader("User-Agent"))
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse request body")
		return
	}
	reqModel, _ := reqBody["model"].(string)
	rawStats := collectRawChatContentStats(reqBody["messages"])

	normalizedReq, convErr := normalizeChatCompletionsRequest(reqBody)
	if convErr != nil {
		if rawStats.RawImageParts > 0 || rawStats.RawInvalidImageParts > 0 || rawStats.RawUnknownParts > 0 {
			log.Printf("[OpenAI ChatCompat] normalization failed: model=%s raw_images=%d invalid_images=%d unknown_parts=%d unknown_types=%s error=%v",
				reqModel,
				rawStats.RawImageParts,
				rawStats.RawInvalidImageParts,
				rawStats.RawUnknownParts,
				rawStats.UnknownTypesString(),
				convErr,
			)
		}
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", convErr.Error())
		return
	}
	normalizedStats := collectNormalizedChatInputStats(normalizedReq["input"])
	if rawStats.RawImageParts > 0 || rawStats.RawUnknownParts > 0 || rawStats.RawInvalidImageParts > 0 {
		log.Printf("[OpenAI ChatCompat] multimodal normalization: model=%s raw_messages=%d raw_images=%d invalid_images=%d raw_unknown_parts=%d unknown_types=%s normalized_input_items=%d normalized_images=%d normalized_text_parts=%d",
			reqModel,
			rawStats.RawMessages,
			rawStats.RawImageParts,
			rawStats.RawInvalidImageParts,
			rawStats.RawUnknownParts,
			rawStats.UnknownTypesString(),
			normalizedStats.InputItems,
			normalizedStats.InputImageParts,
			normalizedStats.InputTextParts,
		)
	}
	if rawStats.RawImageParts > normalizedStats.InputImageParts {
		log.Printf("[OpenAI ChatCompat] image parts dropped during normalization: model=%s raw_images=%d normalized_images=%d dropped=%d",
			reqModel,
			rawStats.RawImageParts,
			normalizedStats.InputImageParts,
			rawStats.RawImageParts-normalizedStats.InputImageParts,
		)
	}

	normalizedBody, err := json.Marshal(normalizedReq)
	if err != nil {
		h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to process request")
		return
	}

	c.Set(service.CtxKeyOpenAIChatCompletionsCompat, true)
	c.Request.Body = io.NopCloser(bytes.NewReader(normalizedBody))
	c.Request.ContentLength = int64(len(normalizedBody))
	c.Request.Header.Set("Content-Type", "application/json")

	h.Responses(c)
}

func normalizeChatCompletionsRequest(req map[string]any) (map[string]any, error) {
	normalized := make(map[string]any, len(req)+2)
	for k, v := range req {
		normalized[k] = v
	}

	// OpenAI chat.completions commonly uses max_tokens/max_completion_tokens.
	// Responses API expects max_output_tokens.
	if _, ok := normalized["max_output_tokens"]; !ok {
		if v, ok := normalized["max_completion_tokens"]; ok {
			normalized["max_output_tokens"] = v
		} else if v, ok := normalized["max_tokens"]; ok {
			normalized["max_output_tokens"] = v
		}
	}

	// Convert chat tools shape to responses tools shape when possible:
	// {"type":"function","function":{"name":"x","parameters":{...}}}
	// =>
	// {"type":"function","name":"x","parameters":{...}}
	if toolsRaw, ok := normalized["tools"].([]any); ok {
		convertedTools := make([]any, 0, len(toolsRaw))
		for _, item := range toolsRaw {
			toolMap, ok := item.(map[string]any)
			if !ok {
				convertedTools = append(convertedTools, item)
				continue
			}
			if toolType, _ := toolMap["type"].(string); toolType == "function" {
				if fn, ok := toolMap["function"].(map[string]any); ok {
					converted := map[string]any{"type": "function"}
					if name, ok := fn["name"]; ok {
						converted["name"] = name
					}
					if desc, ok := fn["description"]; ok {
						converted["description"] = desc
					}
					if params, ok := fn["parameters"]; ok {
						converted["parameters"] = params
					}
					if strict, ok := fn["strict"]; ok {
						converted["strict"] = strict
					}
					convertedTools = append(convertedTools, converted)
					continue
				}
			}
			convertedTools = append(convertedTools, item)
		}
		normalized["tools"] = convertedTools
	}

	// If input already provided, keep it untouched.
	if _, ok := normalized["input"]; ok {
		return normalized, nil
	}

	messagesRaw, ok := normalized["messages"].([]any)
	if !ok || len(messagesRaw) == 0 {
		return nil, fmt.Errorf("messages is required")
	}

	var systemInstructions []string
	inputItems := make([]any, 0, len(messagesRaw))

	for _, raw := range messagesRaw {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "" {
			continue
		}
		content := extractMessageText(msg["content"])
		contentParts := buildResponsesInputContent(msg["content"])
		if role == "system" {
			if strings.TrimSpace(content) != "" {
				systemInstructions = append(systemInstructions, content)
			}
			continue
		}
		if role == "assistant" {
			if toolCallsRaw, ok := msg["tool_calls"].([]any); ok && len(toolCallsRaw) > 0 {
				for i, tcRaw := range toolCallsRaw {
					tc, ok := tcRaw.(map[string]any)
					if !ok {
						continue
					}
					tcType, _ := tc["type"].(string)
					if tcType != "" && tcType != "function" {
						continue
					}
					callID, _ := tc["id"].(string)
					if strings.TrimSpace(callID) == "" {
						callID = fmt.Sprintf("call_%d", i)
					}

					fn, _ := tc["function"].(map[string]any)
					if fn == nil {
						continue
					}
					name, _ := fn["name"].(string)
					if strings.TrimSpace(name) == "" {
						continue
					}

					arguments := ""
					switch v := fn["arguments"].(type) {
					case string:
						arguments = v
					case nil:
						arguments = ""
					default:
						// Keep compatibility with clients that send parsed objects instead of JSON string.
						if b, err := json.Marshal(v); err == nil {
							arguments = string(b)
						}
					}

					inputItems = append(inputItems, map[string]any{
						"type":      "function_call",
						"call_id":   callID,
						"name":      name,
						"arguments": arguments,
					})
				}
				// Some clients send assistant text + tool_calls in the same message.
				// Preserve text as a normal assistant message so downstream context stays intact.
				if hasNonEmptyMessageContent(contentParts) {
					inputItems = append(inputItems, map[string]any{
						"type": "message",
						"role": role,
						"content": contentParts,
					})
				}
				continue
			}
		}
		if role == "tool" {
			item := map[string]any{
				"type":   "function_call_output",
				"output": content,
			}
			if callID, ok := msg["tool_call_id"].(string); ok && strings.TrimSpace(callID) != "" {
				item["call_id"] = callID
			}
			inputItems = append(inputItems, item)
			continue
		}
		inputItems = append(inputItems, map[string]any{
			"type": "message",
			"role": role,
			"content": ensureNonEmptyMessageContent(contentParts, content),
		})
	}

	if len(inputItems) == 0 {
		return nil, fmt.Errorf("messages is required")
	}

	normalized["input"] = inputItems
	if _, ok := normalized["instructions"]; !ok && len(systemInstructions) > 0 {
		normalized["instructions"] = strings.Join(systemInstructions, "\n\n")
	}
	delete(normalized, "messages")

	return normalized, nil
}

func extractMessageText(raw any) string {
	switch v := raw.(type) {
	case string:
		return v
	case []any:
		parts := make([]string, 0, len(v))
		for _, partRaw := range v {
			part, ok := partRaw.(map[string]any)
			if !ok {
				continue
			}
			partType, _ := part["type"].(string)
			switch partType {
			case "text", "input_text", "output_text":
				if text, ok := part["text"].(string); ok {
					parts = append(parts, text)
				}
			default:
				// Keep compatibility with simple multimodal placeholders:
				// ignore non-text segments instead of failing.
			}
		}
		return strings.Join(parts, "")
	default:
		return ""
	}
}

func buildResponsesInputContent(raw any) []map[string]any {
	switch v := raw.(type) {
	case string:
		return []map[string]any{
			{
				"type": "input_text",
				"text": v,
			},
		}
	case []any:
		parts := make([]map[string]any, 0, len(v))
		for _, partRaw := range v {
			part, ok := partRaw.(map[string]any)
			if !ok {
				continue
			}
			partType, _ := part["type"].(string)
			switch partType {
			case "text", "input_text", "output_text":
				if text, ok := part["text"].(string); ok {
					parts = append(parts, map[string]any{
						"type": "input_text",
						"text": text,
					})
				}
			case "image_url":
				url, detail := extractImageURLPart(part["image_url"])
				if strings.TrimSpace(url) == "" {
					continue
				}
				item := map[string]any{
					"type":      "input_image",
					"image_url": url,
				}
				if strings.TrimSpace(detail) != "" {
					item["detail"] = detail
				}
				parts = append(parts, item)
			case "input_image":
				item := map[string]any{"type": "input_image"}
				if imageURL, ok := part["image_url"].(string); ok && strings.TrimSpace(imageURL) != "" {
					item["image_url"] = imageURL
				}
				if detail, ok := part["detail"].(string); ok && strings.TrimSpace(detail) != "" {
					item["detail"] = detail
				}
				if fileID, ok := part["file_id"].(string); ok && strings.TrimSpace(fileID) != "" {
					item["file_id"] = fileID
				}
				if len(item) > 1 {
					parts = append(parts, item)
				}
			default:
				// Ignore unsupported multimodal segments to keep compatibility.
			}
		}
		return parts
	default:
		return nil
	}
}

func extractImageURLPart(raw any) (url string, detail string) {
	switch v := raw.(type) {
	case string:
		return v, ""
	case map[string]any:
		url, _ = v["url"].(string)
		detail, _ = v["detail"].(string)
		return url, detail
	default:
		return "", ""
	}
}

func hasNonEmptyMessageContent(parts []map[string]any) bool {
	for _, part := range parts {
		partType, _ := part["type"].(string)
		switch partType {
		case "input_text":
			if text, ok := part["text"].(string); ok && strings.TrimSpace(text) != "" {
				return true
			}
		case "input_image":
			if imageURL, ok := part["image_url"].(string); ok && strings.TrimSpace(imageURL) != "" {
				return true
			}
			if fileID, ok := part["file_id"].(string); ok && strings.TrimSpace(fileID) != "" {
				return true
			}
		}
	}
	return false
}

func ensureNonEmptyMessageContent(parts []map[string]any, fallbackText string) []map[string]any {
	if len(parts) > 0 {
		return parts
	}
	return []map[string]any{
		{
			"type": "input_text",
			"text": fallbackText,
		},
	}
}

type rawChatContentStats struct {
	RawMessages          int
	RawImageParts        int
	RawInvalidImageParts int
	RawUnknownParts      int
	unknownTypes         map[string]int
}

func (s rawChatContentStats) UnknownTypesString() string {
	if len(s.unknownTypes) == 0 {
		return "-"
	}
	keys := make([]string, 0, len(s.unknownTypes))
	for k := range s.unknownTypes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s:%d", k, s.unknownTypes[k]))
	}
	return strings.Join(parts, ",")
}

type normalizedChatInputStats struct {
	InputItems      int
	InputImageParts int
	InputTextParts  int
}

func collectRawChatContentStats(messages any) rawChatContentStats {
	stats := rawChatContentStats{unknownTypes: make(map[string]int)}
	items, ok := messages.([]any)
	if !ok {
		return stats
	}
	stats.RawMessages = len(items)
	for _, msgRaw := range items {
		msg, ok := msgRaw.(map[string]any)
		if !ok {
			continue
		}
		contentRaw, exists := msg["content"]
		if !exists {
			continue
		}
		switch content := contentRaw.(type) {
		case string:
			// string content has no explicit multimodal part type.
			continue
		case []any:
			for _, partRaw := range content {
				part, ok := partRaw.(map[string]any)
				if !ok {
					stats.RawUnknownParts++
					stats.unknownTypes["non_object"]++
					continue
				}
				partType, _ := part["type"].(string)
				switch partType {
				case "text", "input_text", "output_text":
					continue
				case "image_url":
					stats.RawImageParts++
					url, _ := extractImageURLPart(part["image_url"])
					if strings.TrimSpace(url) == "" {
						stats.RawInvalidImageParts++
					}
				case "input_image":
					stats.RawImageParts++
					url, _ := part["image_url"].(string)
					fileID, _ := part["file_id"].(string)
					if strings.TrimSpace(url) == "" && strings.TrimSpace(fileID) == "" {
						stats.RawInvalidImageParts++
					}
				default:
					stats.RawUnknownParts++
					key := strings.TrimSpace(partType)
					if key == "" {
						key = "unknown"
					}
					stats.unknownTypes[key]++
				}
			}
		default:
			stats.RawUnknownParts++
			stats.unknownTypes["non_array_content"]++
		}
	}
	return stats
}

func collectNormalizedChatInputStats(input any) normalizedChatInputStats {
	stats := normalizedChatInputStats{}
	items, ok := input.([]any)
	if !ok {
		return stats
	}
	stats.InputItems = len(items)
	for _, itemRaw := range items {
		item, ok := itemRaw.(map[string]any)
		if !ok {
			continue
		}
		if itemType, _ := item["type"].(string); itemType != "message" {
			continue
		}
		contentRaw, ok := item["content"]
		if !ok {
			continue
		}
		content, ok := contentRaw.([]map[string]any)
		if !ok {
			continue
		}
		for _, part := range content {
			partType, _ := part["type"].(string)
			switch partType {
			case "input_text":
				stats.InputTextParts++
			case "input_image":
				stats.InputImageParts++
			}
		}
	}
	return stats
}

// handleConcurrencyError handles concurrency-related errors with proper 429 response
func (h *OpenAIGatewayHandler) handleConcurrencyError(c *gin.Context, err error, slotType string, streamStarted bool) {
	h.handleStreamingAwareError(c, http.StatusTooManyRequests, "rate_limit_error",
		fmt.Sprintf("Concurrency limit exceeded for %s, please retry later", slotType), streamStarted)
}

func (h *OpenAIGatewayHandler) handleFailoverExhausted(c *gin.Context, failoverErr *service.UpstreamFailoverError, streamStarted bool) {
	statusCode := failoverErr.StatusCode
	responseBody := failoverErr.ResponseBody

	// 先检查透传规则
	if h.errorPassthroughService != nil && len(responseBody) > 0 {
		if rule := h.errorPassthroughService.MatchRule("openai", statusCode, responseBody); rule != nil {
			// 确定响应状态码
			respCode := statusCode
			if !rule.PassthroughCode && rule.ResponseCode != nil {
				respCode = *rule.ResponseCode
			}

			// 确定响应消息
			msg := service.ExtractUpstreamErrorMessage(responseBody)
			if !rule.PassthroughBody && rule.CustomMessage != nil {
				msg = *rule.CustomMessage
			}

			if rule.SkipMonitoring {
				c.Set(service.OpsSkipPassthroughKey, true)
			}

			h.handleStreamingAwareError(c, respCode, "upstream_error", msg, streamStarted)
			return
		}
	}

	// 使用默认的错误映射
	status, errType, errMsg := h.mapUpstreamError(statusCode)
	h.handleStreamingAwareError(c, status, errType, errMsg, streamStarted)
}

// handleFailoverExhaustedSimple 简化版本，用于没有响应体的情况
func (h *OpenAIGatewayHandler) handleFailoverExhaustedSimple(c *gin.Context, statusCode int, streamStarted bool) {
	status, errType, errMsg := h.mapUpstreamError(statusCode)
	h.handleStreamingAwareError(c, status, errType, errMsg, streamStarted)
}

func (h *OpenAIGatewayHandler) mapUpstreamError(statusCode int) (int, string, string) {
	switch statusCode {
	case 401:
		return http.StatusBadGateway, "upstream_error", "Upstream authentication failed, please contact administrator"
	case 403:
		return http.StatusBadGateway, "upstream_error", "Upstream access forbidden, please contact administrator"
	case 429:
		return http.StatusTooManyRequests, "rate_limit_error", "Upstream rate limit exceeded, please retry later"
	case 529:
		return http.StatusServiceUnavailable, "upstream_error", "Upstream service overloaded, please retry later"
	case 500, 502, 503, 504:
		return http.StatusBadGateway, "upstream_error", "Upstream service temporarily unavailable"
	default:
		return http.StatusBadGateway, "upstream_error", "Upstream request failed"
	}
}

// handleStreamingAwareError handles errors that may occur after streaming has started
func (h *OpenAIGatewayHandler) handleStreamingAwareError(c *gin.Context, status int, errType, message string, streamStarted bool) {
	if streamStarted {
		// Stream already started, send error as SSE event then close
		flusher, ok := c.Writer.(http.Flusher)
		if ok {
			// Send error event in OpenAI SSE format
			errorEvent := fmt.Sprintf(`event: error`+"\n"+`data: {"error": {"type": "%s", "message": "%s"}}`+"\n\n", errType, message)
			if _, err := fmt.Fprint(c.Writer, errorEvent); err != nil {
				_ = c.Error(err)
			}
			flusher.Flush()
		}
		return
	}

	// Normal case: return JSON response with proper status code
	h.errorResponse(c, status, errType, message)
}

// errorResponse returns OpenAI API format error response
func (h *OpenAIGatewayHandler) errorResponse(c *gin.Context, status int, errType, message string) {
	c.JSON(status, gin.H{
		"error": gin.H{
			"type":    errType,
			"message": message,
		},
	})
}
