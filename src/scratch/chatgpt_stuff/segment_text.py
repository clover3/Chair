def merge_segments_to_limit(input_string, length_limit):
    # Split the input string into initial segments
    initial_segments = input_string.split("\n\n")

    # List to hold the final merged segments
    final_segments = []

    current_segment = ""  # Initialize the current segment
    for segment in initial_segments:
        # Prepare the segment by replacing single newlines with spaces to maintain continuity
        segment = segment.replace("\n", " ")

        # Check if adding the new segment exceeds the length limit
        if len(current_segment) + len(segment) + 1 <= length_limit:  # +1 for the space that might be needed
            # If it doesn't exceed, add it to the current segment
            current_segment = segment if current_segment == "" else current_segment + "\n\n" + segment
        else:
            # If it exceeds, append the current segment to the final list and start a new one
            final_segments.append(current_segment)
            current_segment = segment

    # Append the last segment if it's not empty
    if current_segment:
        final_segments.append(current_segment)

    return final_segments

def test_it():
    # Example usage
    input_string = "This is a test string.\n\nThis string is meant to be segmented into smaller parts based on the presence of two consecutive newline characters.\n\nEach segment should not exceed a specified length limit."
    length_limit = 30  # Define the maximum length of each segment
    print(input_string)

    segments = merge_segments_to_limit(input_string, length_limit)
    for i, segment in enumerate(segments):
        print(f"Segment {i + 1} (Length {len(segment)}):\n{segment}\n")
