package handler

import "testing"

func TestNormalizeChatCompletionsRequest_ConvertsAssistantToolCalls(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.2",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"tool_calls": []any{
					map[string]any{
						"id":   "call_abc",
						"type": "function",
						"function": map[string]any{
							"name":      "edit_file",
							"arguments": "{\"path\":\"README.md\"}",
						},
					},
				},
			},
			map[string]any{
				"role":         "tool",
				"tool_call_id": "call_abc",
				"content":      "done",
			},
		},
	}

	normalized, err := normalizeChatCompletionsRequest(req)
	if err != nil {
		t.Fatalf("normalizeChatCompletionsRequest error: %v", err)
	}

	input, ok := normalized["input"].([]any)
	if !ok || len(input) != 2 {
		t.Fatalf("expected 2 input items, got %+v", normalized["input"])
	}

	first, _ := input[0].(map[string]any)
	if first["type"] != "function_call" {
		t.Fatalf("expected function_call item, got %+v", first)
	}
	if first["call_id"] != "call_abc" || first["name"] != "edit_file" {
		t.Fatalf("unexpected function_call fields: %+v", first)
	}

	second, _ := input[1].(map[string]any)
	if second["type"] != "function_call_output" {
		t.Fatalf("expected function_call_output item, got %+v", second)
	}
	if second["call_id"] != "call_abc" || second["output"] != "done" {
		t.Fatalf("unexpected function_call_output fields: %+v", second)
	}
}

func TestNormalizeChatCompletionsRequest_ToolCallArgumentsObject(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.2",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"tool_calls": []any{
					map[string]any{
						"type": "function",
						"function": map[string]any{
							"name": "edit_file",
							"arguments": map[string]any{
								"path": "README.md",
							},
						},
					},
				},
			},
		},
	}

	normalized, err := normalizeChatCompletionsRequest(req)
	if err != nil {
		t.Fatalf("normalizeChatCompletionsRequest error: %v", err)
	}

	input, ok := normalized["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("expected 1 input item, got %+v", normalized["input"])
	}
	first, _ := input[0].(map[string]any)
	if first["type"] != "function_call" {
		t.Fatalf("expected function_call item, got %+v", first)
	}
	if first["call_id"] != "call_0" {
		t.Fatalf("expected synthesized call_0, got %+v", first["call_id"])
	}
	if first["arguments"] != "{\"path\":\"README.md\"}" {
		t.Fatalf("expected marshaled JSON args, got %+v", first["arguments"])
	}
}

func TestNormalizeChatCompletionsRequest_ConvertsImageURLContent(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.2",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "text",
						"text": "请看这张图",
					},
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url":    "https://example.com/cat.png",
							"detail": "high",
						},
					},
				},
			},
		},
	}

	normalized, err := normalizeChatCompletionsRequest(req)
	if err != nil {
		t.Fatalf("normalizeChatCompletionsRequest error: %v", err)
	}

	input, ok := normalized["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("expected 1 input item, got %+v", normalized["input"])
	}

	msg, _ := input[0].(map[string]any)
	if msg["type"] != "message" {
		t.Fatalf("expected message item, got %+v", msg)
	}
	content, ok := msg["content"].([]map[string]any)
	if !ok || len(content) != 2 {
		t.Fatalf("expected 2 content parts, got %+v", msg["content"])
	}
	if content[0]["type"] != "input_text" || content[0]["text"] != "请看这张图" {
		t.Fatalf("unexpected first content part: %+v", content[0])
	}
	if content[1]["type"] != "input_image" || content[1]["image_url"] != "https://example.com/cat.png" || content[1]["detail"] != "high" {
		t.Fatalf("unexpected second content part: %+v", content[1])
	}
}

func TestNormalizeChatCompletionsRequest_ConvertsImageURLString(t *testing.T) {
	req := map[string]any{
		"model": "gpt-5.2",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image_url",
						"image_url": "https://example.com/dog.png",
					},
				},
			},
		},
	}

	normalized, err := normalizeChatCompletionsRequest(req)
	if err != nil {
		t.Fatalf("normalizeChatCompletionsRequest error: %v", err)
	}

	input, ok := normalized["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("expected 1 input item, got %+v", normalized["input"])
	}

	msg, _ := input[0].(map[string]any)
	content, ok := msg["content"].([]map[string]any)
	if !ok || len(content) != 1 {
		t.Fatalf("expected 1 content part, got %+v", msg["content"])
	}
	if content[0]["type"] != "input_image" || content[0]["image_url"] != "https://example.com/dog.png" {
		t.Fatalf("unexpected image content part: %+v", content[0])
	}
}

func TestCollectRawChatContentStats(t *testing.T) {
	messages := []any{
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "hello"},
				map[string]any{"type": "image_url", "image_url": "https://example.com/a.png"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": ""}},
				map[string]any{"type": "input_image", "file_id": "file_123"},
				map[string]any{"type": "input_image"},
				map[string]any{"type": "video_url", "video_url": "https://example.com/v.mp4"},
				42,
			},
		},
		map[string]any{
			"role":    "user",
			"content": "plain text",
		},
		map[string]any{
			"role":    "user",
			"content": 123,
		},
	}

	stats := collectRawChatContentStats(messages)
	if stats.RawMessages != 3 {
		t.Fatalf("expected 3 messages, got %d", stats.RawMessages)
	}
	if stats.RawImageParts != 4 {
		t.Fatalf("expected 4 raw image parts, got %d", stats.RawImageParts)
	}
	if stats.RawInvalidImageParts != 2 {
		t.Fatalf("expected 2 invalid image parts, got %d", stats.RawInvalidImageParts)
	}
	if stats.RawUnknownParts != 3 {
		t.Fatalf("expected 3 unknown parts, got %d", stats.RawUnknownParts)
	}
	if stats.unknownTypes["video_url"] != 1 {
		t.Fatalf("expected video_url unknown count=1, got %+v", stats.unknownTypes)
	}
	if stats.unknownTypes["non_object"] != 1 {
		t.Fatalf("expected non_object unknown count=1, got %+v", stats.unknownTypes)
	}
	if stats.unknownTypes["non_array_content"] != 1 {
		t.Fatalf("expected non_array_content unknown count=1, got %+v", stats.unknownTypes)
	}
}

func TestCollectNormalizedChatInputStats(t *testing.T) {
	input := []any{
		map[string]any{
			"type": "message",
			"content": []map[string]any{
				{"type": "input_text", "text": "hello"},
				{"type": "input_image", "image_url": "https://example.com/a.png"},
			},
		},
		map[string]any{
			"type": "function_call",
		},
		map[string]any{
			"type": "message",
			"content": []map[string]any{
				{"type": "input_image", "file_id": "file_456"},
			},
		},
	}

	stats := collectNormalizedChatInputStats(input)
	if stats.InputItems != 3 {
		t.Fatalf("expected 3 input items, got %d", stats.InputItems)
	}
	if stats.InputTextParts != 1 {
		t.Fatalf("expected 1 input_text part, got %d", stats.InputTextParts)
	}
	if stats.InputImageParts != 2 {
		t.Fatalf("expected 2 input_image parts, got %d", stats.InputImageParts)
	}
}
