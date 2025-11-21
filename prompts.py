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
 
    prompt = f"""
Detected players in this image: {players_str}
 
You see the image with bounding boxes already drawn around each detected player.
Create a descriptive tag for this frame following cricket training video tagging conventions.
 
Rules:
- Maximum 5 words in title case.
- Format: [Player Names] + [Activity/Action]
- Use commas between player names (e.g., "Dhoni, Raina, Rayudu Interaction")
- Describe the primary visible action or activity (e.g., "Batting", "Catching Practice", "Warm Up", "Interaction")
- Keep it concise and searchable.
- Only tag what is clearly visible in the frame.
 
Examples:
- "Dhoni, Raina Nets Interaction"
- "Rayudu Batting"
- "Kohli, Sharma Catching Drill"
 
Return only the tag.
"""
 
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
 
Collate these frame observations into a single descriptive tag. Keep it crisp and concise
 
Rules:
 
- Identify the common players across frames and list them
- Identify the primary activity or theme (e.g., "Interaction", "Nets", "Catching Practice", "Relaxing", "Batting")
- If players are doing the same activity throughout, use that activity
- If activity varies but players interact, use "Interaction"
- If multiple distinct activities occur, choose the dominant one
- Keep it concise and searchable
Return only the final collated tag."""
 
    return prompt
 
