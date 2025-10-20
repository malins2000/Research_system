import re
import json

# --- Helper Function (You can put this in a utility file or at the top of each agent) ---
def parse_llm_json_output(response_str: str, xml_tag: str) -> dict | list | None:
    """
    Tries to parse JSON from LLM output using three stages:
    1. Find specific XML tag, then find the first JSON object/array within that tag.
    2. Fallback: Find the first JSON object/array anywhere in the full response.
    3. Failure: Return None if no valid JSON is found.
    """
    json_str = ""
    parsed_json = None
    stage1_success = False

    # Stage 1: Find XML tag, then find JSON within the tag's content
    xml_match = re.search(rf"<{xml_tag}>(.*?)</{xml_tag}>", response_str, re.DOTALL)
    if xml_match:
        content_within_tags = xml_match.group(1).strip()
        # Now, find the first JSON object '{...}' or array '[...]' INSIDE the tags
        json_inner_match = re.search(r"\{.*\}|\[.*\]", content_within_tags, re.DOTALL)
        if json_inner_match:
            json_str = json_inner_match.group(0).strip()
            try:
                parsed_json = json.loads(json_str)
                print(f"Parsing successful (Stage 1: XML Tag '{xml_tag}' + Inner JSON)")
                stage1_success = True
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Stage 1 Error: JSONDecodeError within <{xml_tag}> tags: {e}")
                # Log the specific string that failed if needed
                # print(f"Stage 1 Failed String: {json_str}")
                # Fall through to Stage 2
        else:
             print(f"Stage 1 Warning: Found <{xml_tag}> tags, but no JSON object/array inside.")
             # Fall through to Stage 2
    else:
        print(f"Stage 1 Info: Could not find <{xml_tag}> tags.")
        # Fall through to Stage 2

    # Stage 2 (Fallback): Find the first JSON object or array anywhere in the full response
    if not stage1_success: # Only run if Stage 1 failed
        print(f"Falling back to Stage 2 (raw JSON search) for tag <{xml_tag}>.")
        json_fallback_match = re.search(r"\{.*\}|\[.*\]", response_str, re.DOTALL)
        if json_fallback_match:
            json_str = json_fallback_match.group(0).strip()
            try:
                parsed_json = json.loads(json_str)
                print("Parsing successful (Stage 2: Fallback Raw JSON Search)")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Stage 2 Error: JSONDecodeError in fallback search: {e}")
                # Log the specific string that failed if needed
                # print(f"Stage 2 Failed String: {json_str}")
                # Fall through to Stage 3 (Failure)
        else:
            print("Stage 2 Info: No JSON object/array found in fallback search.")
            # Fall through to Stage 3 (Failure)

    # Stage 3: Failure
    print(f"Parsing failed after all stages for tag <{xml_tag}>.")
    print(f"Raw LLM Response was:\n{response_str}")
    return None

# --- How to use it inside an agent's execute method ---
# (Example for PlannerAgent)

# response_str = self.llm_client.query(prompt)
# parsed_result = parse_llm_json_output(response_str, "plan_json") # Use the correct tag

# if parsed_result:
#     initial_structure = parsed_result
#     plan_manager.create_plan(prompt=user_prompt, initial_structure=initial_structure)
#     print("Planner Agent: Successfully parsed plan and saved.")
# else:
#     # Handle parsing failure robustly
#     print(f"Planner Agent: CRITICAL - Failed to parse valid JSON from LLM after all fallbacks.")
#     raise ValueError("Failed to parse a valid plan from the LLM. Stopping workflow.")