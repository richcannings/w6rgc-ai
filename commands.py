import re

def handle_command(operator_text, prompt_mgr):
    """
    Identifies known commands from the operator_text.

    Args:
        operator_text (str): The transcribed text from the operator.
        prompt_mgr (PromptManager): The prompt manager instance.

    Returns:
        str: Command type ("terminate", "status", "reset", "identify") or None if no command is matched.
    """
    bot_name = prompt_mgr.get_bot_name() # Get bot_name once

    # Check for termination command
    if re.search(rf"{re.escape(bot_name)}.*?\b(break|brake|exit|quit|shutdown)\b", operator_text, re.IGNORECASE):
        print("🛑 Termination command detected by commands.py.")
        return "terminate"

    # Check for status command
    if re.search(rf"{re.escape(bot_name)}.*?\b(status|report)\b", operator_text, re.IGNORECASE):
        print("⚙️ Status command detected by commands.py.")
        return "status"

    # Check for reset/new chat command
    if re.search(rf"{re.escape(bot_name)}.*?\b(reset|start a new chat|new chat)\b", operator_text, re.IGNORECASE):
        print("🔄 Reset command detected by commands.py.")
        return "reset"

    # Check for identify command
    if re.search(rf"{re.escape(bot_name)}.*?\b(identify)\b", operator_text, re.IGNORECASE) or \
       re.search(r"\b(identify|call sign|what is your call sign|who are you)\b", operator_text, re.IGNORECASE):
        print("🆔 Identify command detected by commands.py.")
        return "identify"
        
    return None 