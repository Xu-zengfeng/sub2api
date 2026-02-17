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
