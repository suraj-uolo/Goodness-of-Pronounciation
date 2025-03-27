import json
import sys
import os
import logging
import re
import numpy as np
from main import run_gop

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_phone_map(phones_file):
    """Load phone ID to phone symbol mapping"""
    phone_map = {}
    try:
        with open(phones_file, 'r') as f:
            for line in f:
                phone, phone_id = line.strip().split()
                phone_map[int(phone_id)] = phone
    except Exception as e:
        logging.error(f"Error loading phone map: {e}")
    return phone_map

def validate_inputs(wav_file, transcript):
    """Validate input files and parameters"""
    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"WAV file not found: {wav_file}")
    if not transcript or not isinstance(transcript, str):
        raise ValueError("Invalid transcript")
    if not transcript.strip():
        raise ValueError("Transcript cannot be empty")

def normalize_score(score):
    """Normalize score to be between 0 and 1"""
    try:
        return min(max(float(score), 0), 1)
    except (ValueError, TypeError):
        logging.warning(f"Invalid score value: {score}, defaulting to 0")
        return 0

def format_phone_score(phone_data, phone_map):
    """Format individual phone score data"""
    try:
        phone_id, phone, score, prob = phone_data
        phone_symbol = phone_map.get(int(phone_id), f"PHONE_{phone_id}")
        normalized_prob = normalize_score(prob)
        
        # Debug logging
        logging.info(f"Phone score: ID={phone_id}, Symbol={phone_symbol}, Score={normalized_prob}")
        
        return {
            "phone_id": str(phone_id),
            "phone": phone_symbol,
            "quality_score": normalized_prob,
            "stress_level": None,  # Not available in current scope
            "sound_most_like": phone_symbol
        }
    except (IndexError, ValueError) as e:
        logging.error(f"Error formatting phone score: {e}")
        return None

def segment_transcript(transcript):
    """Segment transcript into words"""
    return [word.strip() for word in transcript.split() if word.strip()]

def format_word_score(word, phone_scores, phone_map):
    """Format word level score data"""
    try:
        # Filter out None values and calculate average quality score
        valid_scores = [ps for ps in phone_scores if ps is not None]
        if not valid_scores:
            logging.warning(f"No valid phone scores for word: {word}")
            return None
            
        # Debug logging
        logging.info(f"Processing word: {word}")
        logging.info(f"Valid phone scores: {len(valid_scores)}")
        
        quality_scores = [normalize_score(ps[3]) for ps in valid_scores]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        word_score = {
            "word": word,
            "quality_score": round(avg_quality * 100),
            "phone_score_list": []
        }
        
        # Add phone scores
        for ps in valid_scores:
            phone_score = format_phone_score(ps, phone_map)
            if phone_score is not None:
                word_score["phone_score_list"].append(phone_score)
        
        # Debug logging
        logging.info(f"Word score created with {len(word_score['phone_score_list'])} phones")
        
        return word_score
    except Exception as e:
        logging.error(f"Error processing word {word}: {e}")
        return None

def format_phone_durations(phone_durations, phone_map):
    """Format and validate phone durations"""
    try:
        formatted_durations = []
        for duration in phone_durations:
            if len(duration) != 3:
                logging.warning(f"Invalid duration format: {duration}")
                continue
            phone_id, start, end = duration
            phone_symbol = phone_map.get(int(phone_id), f"PHONE_{phone_id}")
            formatted_durations.append([
                phone_symbol,
                float(start),
                float(end)
            ])
        return formatted_durations
    except Exception as e:
        logging.error(f"Error formatting phone durations: {e}")
        return []

def format_gop_output(transcript, phone_scores, phone_durations, avg_score, phone_map):
    """Format GOP output with proper word segmentation"""
    try:
        # Split transcript into words
        words = segment_transcript(transcript)
        
        # Create a mapping of phone scores by phone ID
        phone_scores_by_id = {ps[0]: ps for ps in phone_scores}
        
        # Debug logging
        logging.info(f"Transcript words: {words}")
        logging.info(f"Number of phone scores: {len(phone_scores)}")
        logging.info(f"Phone scores by ID: {phone_scores_by_id}")
        
        # Create the formatted output
        output = {
            "status": "success",
            "speech_score": {
                "transcript": transcript,
                "word_score_list": [],
                "overall_metrics": {
                    "pronunciation_score": round(normalize_score(avg_score) * 100),
                    "phone_durations": format_phone_durations(phone_durations, phone_map)
                }
            }
        }
        
        # Read phone alignments
        ali_file = 'exp/gop_test_api/ali-phone.1'
        with open(ali_file, 'r') as f:
            ali_line = f.readline().strip()
            
        # Parse phone alignments
        phone_sequence = []
        for pid in ali_line.split()[1:]:
            try:
                phone_id = int(pid)
                if phone_id in phone_map:
                    phone_sequence.append((phone_id, phone_map[phone_id]))
                else:
                    logging.warning(f"Unknown phone ID in alignment: {phone_id}")
            except ValueError:
                logging.warning(f"Invalid phone ID in alignment: {pid}")
        
        logging.info(f"Phone sequence: {phone_sequence}")
        
        # Process each word
        current_phone_idx = 0
        for word in words:
            word_phones = []
            # Approximate number of phones for this word (rough estimate)
            num_phones = len(word)
            
            # Get phones for this word from the sequence
            while current_phone_idx < len(phone_sequence) and len(word_phones) < num_phones:
                phone_id, phone_symbol = phone_sequence[current_phone_idx]
                
                # Skip silence phones
                if phone_id == 1:  # SIL
                    current_phone_idx += 1
                    continue
                    
                # Get the corresponding score if available
                if phone_id in phone_scores_by_id:
                    word_phones.append(phone_scores_by_id[phone_id])
                else:
                    logging.warning(f"No score found for phone {phone_symbol} (ID: {phone_id})")
                
                current_phone_idx += 1
            
            if word_phones:
                word_score = format_word_score(word, word_phones, phone_map)
                if word_score is not None:
                    output["speech_score"]["word_score_list"].append(word_score)
                    logging.info(f"Added word score for '{word}' with {len(word_phones)} phones")
        
        # Debug logging
        logging.info(f"Number of words with scores: {len(output['speech_score']['word_score_list'])}")
        
        return output
    except Exception as e:
        logging.error(f"Error formatting GOP output: {e}")
        return None

def main():
    try:
        if len(sys.argv) != 3:
            print("Usage: python gop_formatter.py <wav_file> <transcript>")
            sys.exit(1)
            
        wav_file = sys.argv[1]
        transcript = sys.argv[2]
        
        # Load phone mapping
        phones_file = 'exp/gop_test_api/phones-pure.txt'
        phone_map = load_phone_map(phones_file)
        if not phone_map:
            logging.error("Failed to load phone mapping")
            sys.exit(1)
            
        # Validate inputs
        validate_inputs(wav_file, transcript)
        
        # Read GOP scores directly from Kaldi output
        gop_file = 'exp/gop_test_api/gop.1.txt'
        if not os.path.exists(gop_file):
            logging.error(f"GOP file not found: {gop_file}")
            sys.exit(1)
            
        with open(gop_file, 'r') as f:
            gop_line = f.readline().strip()
            
        if not gop_line:
            logging.error("Empty GOP file")
            sys.exit(1)
            
        # Parse GOP scores
        phone_scores = []
        for match in re.findall(r'\[(\d+)\s+([-\d.]+)\]', gop_line):
            phone_id, score = match
            phone_id = int(phone_id)
            score = float(score)
            prob = np.power(10, score)
            
            # Validate phone ID exists in mapping
            if phone_id not in phone_map:
                logging.warning(f"Unknown phone ID: {phone_id}")
                continue
                
            phone_scores.append((phone_id, phone_map[phone_id], score, prob))
            
        if not phone_scores:
            logging.error("No valid phone scores found")
            sys.exit(1)
            
        logging.info(f"Found {len(phone_scores)} phone scores")
        
        # Calculate average score
        valid_scores = [score for _, _, _, score in phone_scores if score > 0]
        if not valid_scores:
            logging.error("No valid scores found")
            sys.exit(1)
            
        avg_score = sum(valid_scores) / len(valid_scores)
        logging.info(f"Average score: {avg_score}")
        
        # Format the output
        formatted_output = format_gop_output(
            transcript,
            phone_scores,
            [],  # Phone durations not available in current scope
            avg_score,
            phone_map
        )
        
        if formatted_output is None:
            logging.error("Failed to format GOP output")
            sys.exit(1)
            
        # Validate output structure
        if not formatted_output.get("speech_score", {}).get("word_score_list"):
            logging.error("No word scores in output")
            sys.exit(1)
        
        # Write to output file
        output_file = os.path.splitext(wav_file)[0] + '_gop.json'
        with open(output_file, 'w') as f:
            json.dump(formatted_output, f, indent=2)
        
        logging.info(f"Results written to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 