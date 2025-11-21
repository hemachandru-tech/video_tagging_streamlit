"""
Simplified prompt templates for cricket frame analysis.
LLM sees the frame with bounding boxes + detected player names.
Return minimal, direct, player-specific captions.
"""
 
def get_frame_analysis_prompt(player_names):
    """
    Prompt for analyzing a single cricket frame.
    Ensures LLM clearly describes visible actions of the detected players.
    """
    players_str = ", ".join(player_names)
 
    prompt = f"""Detected players in this image: {players_str}
 
You see the image with bounding boxes already drawn around each detected player.
Describe exactly what each detected player is doing right now.
 
Rules:
- Maximum 20 words.
- Keep it simple and factual.
- Mention player names.
- Only describe visible body posture, action, equipment, and mood.
- No assumptions or guesses. No extra details.
- Only describe what is clearly visible in the frame.
 
Return only a short caption."""
 
    return prompt
 
def get_final_summary_prompt(frame_analyses, all_detected_players):
    """
    Generate a single summary of exactly 15 words
    based ONLY on visible frame descriptions.
    """
    analysis_context = ""
    for analysis_data in frame_analyses:
        frame_num = analysis_data['frame']
        timestamp = analysis_data['timestamp']
        players = ", ".join(analysis_data['players'])
        analysis = analysis_data['analysis']
 
        analysis_context += f"Frame {frame_num} ({timestamp:.1f}s) - {players}: {analysis}\n"
 
    players_str = ", ".join(all_detected_players)
 
    prompt = f"""You reviewed multiple frames showing these players: {players_str}
 
Frame observations:
{analysis_context}
 
Now create ONE final summary describing the overall action visible in these frames.
 
Rules:
- The summary MUST be exactly 15 words.
- Use simple, clear language.
- Describe only visible actions, posture, movement, mood, and location.
- No assumptions or invented details.
- Combine the meaning of all frame captions into one natural 15-word description.
 
Return only the 15-word summary and nothing else."""
 
    return prompt
 
 