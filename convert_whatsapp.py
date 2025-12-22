import re
import csv
from datetime import datetime

def parse_whatsapp_to_csv(input_file, output_file):
    """
    Convert WhatsApp export file to CSV format.
    Filters out messages by 'Mian Ali Ahmed'.
    Handles multi-line messages.
    """
    # Pattern to match WhatsApp message format: DD/MM/YYYY, HH:MM - Author: Message
    message_pattern = re.compile(r'^(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}) - (.+?): (.+)$')
    
    messages = []
    current_message = None
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Try to match the message pattern
            match = message_pattern.match(line)
            
            if match:
                # New message found
                date_str, time_str, author, message_text = match.groups()
                
                # Skip messages by Mian Ali Ahmed
                if author.strip() == "Mian Ali Ahmed":
                    current_message = None
                    continue
                
                # Combine date and time into a single timestamp
                timestamp = f"{date_str} {time_str}"
                
                # Create new message entry
                current_message = {
                    'timestamp': timestamp,
                    'message': message_text
                }
                messages.append(current_message)
            else:
                # Continuation line (part of previous message)
                if current_message is not None:
                    # Append to the current message
                    current_message['message'] += ' ' + line
                # If current_message is None, it means the previous message was from Mian Ali Ahmed
                # or it's a system message, so we skip it
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for msg in messages:
            writer.writerow(msg)
    
    print(f"Conversion complete! {len(messages)} messages saved to {output_file}")
    print(f"(Messages by Mian Ali Ahmed have been filtered out)")

if __name__ == "__main__":
    parse_whatsapp_to_csv('combined.txt', 'messages.csv')

