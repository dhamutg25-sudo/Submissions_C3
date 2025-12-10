"""
YouTube Video Processing Pipeline Tutorial
Converted from Jupyter Notebook
"""


# ======================================================================
# # YouTube Video Processing Pipeline Tutorial
#
# This tutorial demonstrates how to build a complete YouTube video processing pipeline that:
# 1. **Searches** for YouTube videos based on a query
# 2. **Fetches** transcripts from the videos using yt-dlp
# 3. **Summarizes** the transcripts using AI via OpenRouter
#
# ## What You'll Learn
#
# - How to use the YouTube Data API to search for videos
# - How to extract transcripts using yt-dlp
# - How to create AI-powered summaries using OpenRouter
# - How to build a complete automated pipeline
#
# ## Prerequisites
#
# You'll need:
# - YouTube Data API key (free from Google Cloud Console)
# - OpenRouter API key (cheaper alternative to OpenAI)
#
# ## Pipeline Architecture
#
# ```
# Search Query → YouTube API → Video URLs → Transcript Fetcher → AI Summarizer → Final Summaries
# ```
# ======================================================================

# ======================================================================
# ## 1. Environment Setup and Configuration
#
# Install required packages and configure API keys for the YouTube pipeline.
# ======================================================================

# !pip install -r "../requirements.txt"

# Install required packages (run this first)
import subprocess
import sys

print("\n📦 Importing libraries...")
import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from googleapiclient.discovery import build
from yt_dlp import YoutubeDL
from openai import OpenAI

print("✅ All imports successful!")

# https://console.cloud.google.com/
YOUTUBE_API_KEY = ""  # Get from Google Cloud Console
OPENROUTER_API_KEY = "" # Get from OpenRouter

# Configuration parameters
CONFIG = {
    "youtube": {
        "max_results": 3,          # Number of videos to fetch
        "api_version": "v3",       # YouTube API version
        "order": "relevance"       # Search order
    },
    "transcripts": {
        "language": "en",          # Transcript language
        "format": "srt"            # Subtitle format
    },
    "openrouter": {
        "model": "openai/gpt-4o-mini",  # OpenRouter model (much cheaper than direct OpenAI)
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 120             # API timeout in seconds
    },
    "output": {
        "folder": "youtube_pipeline_output"
    }
}

# Create output directories
output_folder = CONFIG["output"]["folder"]
os.makedirs(output_folder, exist_ok=True)
os.makedirs(f"{output_folder}/transcripts", exist_ok=True)
os.makedirs(f"{output_folder}/summaries", exist_ok=True)
os.makedirs(f"{output_folder}/metadata", exist_ok=True)

print(f"✅ Configuration loaded!")
print(f"📁 Output folder: {output_folder}")

# Verify API keys are set
if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
    print("⚠️  Please set your YouTube API key in the YOUTUBE_API_KEY variable")
if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
    print("⚠️  Please set your OpenRouter API key in the OPENROUTER_API_KEY variable")
    print("💡 Get your OpenRouter key at: https://openrouter.ai/keys")

# ======================================================================
# ## 2. YouTube Video Search
#
# Search for YouTube videos using the YouTube Data API with duration parsing.
# ======================================================================

def parse_duration(iso_duration: str) -> str:
    """Parse ISO 8601 duration format (PT4M13S) to human-readable format (4:13).
    
    Args:
        iso_duration (str): ISO 8601 duration string (e.g., "PT4M13S")
        
    Returns:
        str: Human-readable duration (e.g., "4:13")
    """
    if not iso_duration:
        return "Unknown"
    
    # Parse ISO 8601 duration format
    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match = re.match(pattern, iso_duration)
    
    if not match:
        return "Unknown"
    
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    
    # Format duration
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def search_youtube_videos(search_query: str, max_results: int = 3) -> List[Dict]:
    """Search for YouTube videos using the YouTube Data API.
    
    Args:
        search_query (str): The search query to find relevant YouTube videos
        max_results (int): Maximum number of results to return
        
    Returns:
        List[Dict]: List of video information dictionaries
    """
    print(f"🔍 Searching for videos: '{search_query}'")
    
    try:
        # Build the YouTube API client
        youtube = build("youtube", CONFIG["youtube"]["api_version"], developerKey=YOUTUBE_API_KEY)
        
        # Search for videos
        search_request = youtube.search().list(
            q=search_query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            order=CONFIG["youtube"]["order"]
        )
        
        search_response = search_request.execute()
        
        # Extract video IDs and basic info
        video_ids = []
        videos_data = []
        
        for search_result in search_response.get("items", []):
            if "id" in search_result and "videoId" in search_result["id"]:
                video_id = search_result["id"]["videoId"]
                video_ids.append(video_id)
                videos_data.append(search_result)
        
        # Get detailed video information including duration
        videos = []
        if video_ids:
            video_details_request = youtube.videos().list(
                part="contentDetails,statistics", 
                id=",".join(video_ids)
            )
            video_details_response = video_details_request.execute()
            
            # Create a mapping of video_id to details
            video_details_map = {}
            for video_detail in video_details_response.get("items", []):
                video_details_map[video_detail["id"]] = video_detail
        
        # Build final video information
        for i, search_result in enumerate(videos_data):
            video_id = video_ids[i]
            
            # Get duration from video details
            duration = "Unknown"
            if video_id in video_details_map:
                duration_iso = video_details_map[video_id]["contentDetails"]["duration"]
                duration = parse_duration(duration_iso)
            
            description = search_result["snippet"]["description"]
            if len(description) > 200:
                description = description[:200] + "..."
            
            video_info = {
                "title": search_result["snippet"]["title"],
                "channel": search_result["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": description,
                "published_at": search_result["snippet"]["publishedAt"],
                "video_id": video_id,
                "duration": duration
            }
            videos.append(video_info)
        
        print(f"✅ Found {len(videos)} videos")
        return videos
        
    except Exception as e:
        print(f"❌ Error searching YouTube videos: {str(e)}")
        return []

# Test the search function
test_query = "CrewAI tutorial"
print(f"Testing search with query: '{test_query}'")

if YOUTUBE_API_KEY != "YOUR_YOUTUBE_API_KEY_HERE":
    videos = search_youtube_videos(test_query, max_results=2)
    
    # Display results
    print("\n📋 Search Results:")
    for i, video in enumerate(videos, 1):
        print(f"\n{i}. **{video['title']}**")
        print(f"   Channel: {video['channel']}")
        print(f"   Duration: {video['duration']}")
        print(f"   URL: {video['url']}")
else:
    print("⚠️  Please set your YouTube API key to test the search function")

# ======================================================================
# ## 3. Transcript Fetching
#
# Fetch video transcripts using yt-dlp with proper error handling.
# ======================================================================

class YouTubeTranscriptFetcher:
    """A class to fetch transcripts from YouTube videos using yt-dlp."""
    
    def __init__(self, output_folder: str = "transcripts", language: str = "en"):
        """Initialize the transcript fetcher.
        
        Args:
            output_folder (str): Directory where transcripts will be saved
            language (str): Language code for subtitles
        """
        self.output_folder = output_folder
        self.language = language
        os.makedirs(output_folder, exist_ok=True)
    
    def _get_ydl_options(self) -> dict:
        """Get the yt-dlp options configuration.
        
        Returns:
            dict: Configuration options for yt-dlp
        """
        return {
            "skip_download": True,                    # Don't download video
            "writesubtitles": True,                   # Download human captions
            "writeautomaticsub": True,                # Download auto captions
            "subtitleslangs": [self.language],        # Language preference
            "subtitlesformat": "srt",                 # Format preference
            "outtmpl": os.path.join(self.output_folder, "%(id)s.%(ext)s"),
            "ignoreerrors": False,                    # Don't ignore errors
        }
    
    def fetch_transcript(self, url: str) -> bool:
        """Fetch transcript for a single YouTube video.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            bool: True if transcript was successfully downloaded, False otherwise
        """
        try:
            print(f"📥 Fetching transcript for: {url}")
            with YoutubeDL(self._get_ydl_options()) as ydl:
                ydl.download([url])
            print(f"✅ Transcript downloaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error downloading transcript: {str(e)}")
            return False
    
    def fetch_transcripts(self, urls: List[str]) -> Dict[str, bool]:
        """Fetch transcripts for multiple YouTube videos.
        
        Args:
            urls (List[str]): List of YouTube video URLs
            
        Returns:
            Dict[str, bool]: Dictionary with URLs as keys and success status as values
        """
        if not urls:
            return {}
        
        print(f"\n📥 Fetching transcripts for {len(urls)} videos...")
        results = {}
        
        for url in urls:
            results[url] = self.fetch_transcript(url)
        
        successful = sum(results.values())
        print(f"\n✅ Successfully fetched {successful}/{len(urls)} transcripts")
        return results
    
    def get_transcript_files(self, video_ids: List[str]) -> List[str]:
        """Get list of existing transcript files for given video IDs.
        
        Args:
            video_ids (List[str]): List of video IDs
            
        Returns:
            List[str]: List of existing transcript file paths
        """
        transcript_files = []
        
        for video_id in video_ids:
            transcript_path = os.path.join(self.output_folder, f"{video_id}.{self.language}.srt")
            if os.path.exists(transcript_path):
                transcript_files.append(transcript_path)
                print(f"📄 Found transcript: {transcript_path}")
            else:
                print(f"❓ Missing transcript: {transcript_path}")
        
        return transcript_files

# Test the transcript fetcher (only if we have search results)
print("Setting up transcript fetcher...")
transcript_fetcher = YouTubeTranscriptFetcher(
    output_folder=f"{CONFIG['output']['folder']}/transcripts",
    language=CONFIG["transcripts"]["language"]
)

print("✅ Transcript fetcher ready!")
print("Note: Transcript fetching will be demonstrated in the full pipeline section.")

# ======================================================================
# ## 4. AI-Powered Transcript Summarization
#
# Create structured summaries using OpenRouter with a professional summarization prompt.
# ======================================================================

# Define the prompt for video summarization
SUMMARIZER_PROMPT = """
You are **TechSummarizerAI**, an expert AI assistant specializing in analyzing technical videos for software engineers, machine learning engineers, and other technical audiences.

Your goal is to produce a **comprehensive, technical, and structured summary** highlighting key engineering insights, tools, frameworks, system designs, workflows, and implementation processes from the video.

## Objectives
1. **Engineer's Perspective** — Capture technical details over general narration.
2. **Implementation Relevance** — Show *how* the video's concepts can be applied in real-world engineering.
3. **Precision** — Summarize strictly from provided inputs; no speculation.
4. **Clarity** — Maintain concise, professional language.

## Output JSON Schema

Your response must be a valid JSON object with this structure:

{
  "high_level_overview": "String — One paragraph capturing the essence of the video from an engineering viewpoint.",
  "technical_breakdown": [
    {
      "type": "tool", 
      "name": "String — Tool, framework, package, or API name",
      "purpose": "String — Purpose or role in workflow"
    },
    {
      "type": "architecture",
      "description": "String — Detailed architecture or system design notes"
    },
    {
      "type": "process",
      "step_number": "Integer — Step order",
      "description": "String — Process step description"
    }
  ],
  "insights": [
    "String — Key engineering insight, trade-off, or optimization"
  ],
  "applications": [
    "String — Practical application scenario"
  ],
  "limitations": [
    "String — Known limitation, caveat, or risk"
  ]
}

## Formatting Rules

- CRITICAL: Only produce the raw JSON object — no markdown code blocks, no extra text, no ```json wrapper.
- Your response must start with { and end with } as valid JSON.
- Keep text in complete, professional sentences; no fragments.
- Arrays must contain at least one entry if relevant information is available; omit empty arrays.
"""

class YouTubeTranscriptSummarizer:
    """A class to summarize YouTube SRT transcript files using OpenRouter."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the transcript summarizer.
        
        Args:
            api_key (Optional[str]): OpenRouter API key. If None, uses OPENROUTER_API_KEY
        """
        self.client = OpenAI(
            api_key=api_key or OPENROUTER_API_KEY,
            base_url=CONFIG["openrouter"]["base_url"]
        )
        self.model = CONFIG["openrouter"]["model"]
        self.timeout = CONFIG["openrouter"]["timeout"]
    
    def _read_srt_file(self, srt_path: str) -> str:
        """Read and parse SRT file content.
        
        Args:
            srt_path (str): Path to the SRT file
            
        Returns:
            str: The content of the SRT file as plain text
        """
        print(f"📖 Reading SRT file: {os.path.basename(srt_path)}")
        
        with open(srt_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Basic SRT parsing - extract just the text content
        lines = content.split("\n")
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip sequence numbers, timestamps, and empty lines
            if line and not line.isdigit() and "-->" not in line:
                text_lines.append(line)
        
        transcript_text = " ".join(text_lines)
        print(f"📊 Extracted {len(text_lines)} text lines, {len(transcript_text)} characters total")
        return transcript_text
    
    def summarize_transcript(
        self, 
        srt_path: str, 
        video_title: str = "", 
        video_description: str = "", 
        output_path: Optional[str] = None
    ) -> str:
        """Summarize a YouTube SRT transcript file.
        
        Args:
            srt_path (str): Path to the SRT file to summarize
            video_title (str): Title of the video (optional)
            video_description (str): Description of the video (optional)
            output_path (Optional[str]): Path to save the summary. If None, returns summary
            
        Returns:
            str: The generated summary
        """
        print(f"\n🤖 Starting summarization for: {os.path.basename(srt_path)}")
        
        # Read the SRT file
        transcript_text = self._read_srt_file(srt_path)
        
        # Prepare the user message with video details and transcript
        user_message = f"""
**YouTube Video Title:** {video_title if video_title else "Not provided"}

**YouTube Video Description:** {video_description if video_description else "Not provided"}

**Full Transcript:**
{transcript_text}
"""
        
        try:
            print(f"🔄 Making API call to OpenRouter...")
            
            # Make API call to OpenRouter
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARIZER_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                timeout=self.timeout
            )
            
            summary = response.choices[0].message.content
            
            # Save or return the summary
            if output_path:
                print(f"💾 Saving summary to: {os.path.basename(output_path)}")
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(summary)
            
            print(f"✅ Summarization completed")
            return summary
            
        except Exception as e:
            print(f"❌ Error during summarization: {str(e)}")
            return ""
    
    def summarize_transcripts(
        self, 
        transcript_paths: List[str], 
        videos: List[Dict], 
        output_folder: str
    ) -> Dict[str, bool]:
        """Summarize multiple transcript files.
        
        Args:
            transcript_paths (List[str]): List of SRT file paths
            videos (List[Dict]): List of video information for context
            output_folder (str): Folder to save summaries
            
        Returns:
            Dict[str, bool]: Dictionary with file paths as keys and success status as values
        """
        if not transcript_paths:
            return {}
        
        print(f"\n🤖 Starting batch summarization for {len(transcript_paths)} transcripts")
        
        # Create a mapping of video IDs to video info
        video_info_map = {video["video_id"]: video for video in videos}
        
        results = {}
        
        for transcript_path in transcript_paths:
            # Extract video ID from filename
            filename = os.path.basename(transcript_path)
            video_id = filename.split(".")[0]
            video_info = video_info_map.get(video_id, {})
            
            # Create output path
            summary_filename = f"{video_id}_summary.json"
            summary_path = os.path.join(output_folder, summary_filename)
            
            # Summarize transcript
            summary = self.summarize_transcript(
                transcript_path,
                video_info.get("title", ""),
                video_info.get("description", ""),
                summary_path
            )
            
            results[transcript_path] = bool(summary)
        
        successful = sum(results.values())
        print(f"\n✅ Successfully summarized {successful}/{len(transcript_paths)} transcripts")
        return results

# Initialize the summarizer
print("Setting up transcript summarizer...")

if OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
    summarizer = YouTubeTranscriptSummarizer()
    print("✅ Transcript summarizer ready!")
else:
    print("⚠️  Please set your OpenRouter API key to use the summarizer")
    summarizer = None

# ======================================================================
# ## 5. Complete Pipeline Integration
#
# Integrate all components into a single automated pipeline with comprehensive error handling.
# ======================================================================

class YouTubePipeline:
    """Complete pipeline for YouTube video processing.
    
    This class integrates video search, transcript fetching, and summarization
    into a single automated workflow.
    """
    
    def __init__(self, output_folder: str = "youtube_pipeline_output"):
        """Initialize the YouTube pipeline.
        
        Args:
            output_folder (str): Base output folder for all results
        """
        self.output_folder = output_folder
        
        # Create output directories
        self.transcripts_folder = os.path.join(output_folder, "transcripts")
        self.summaries_folder = os.path.join(output_folder, "summaries")
        self.metadata_folder = os.path.join(output_folder, "metadata")
        
        for folder in [self.transcripts_folder, self.summaries_folder, self.metadata_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Initialize components
        self.transcript_fetcher = YouTubeTranscriptFetcher(
            output_folder=self.transcripts_folder,
            language=CONFIG["transcripts"]["language"]
        )
        
        if OPENROUTER_API_KEY != "YOUR_OPENROUTER_API_KEY_HERE":
            self.summarizer = YouTubeTranscriptSummarizer()
        else:
            self.summarizer = None
        
        print(f"🚀 Pipeline initialized")
        print(f"📁 Output folder: {self.output_folder}")
    
    def run_pipeline(self, search_query: str, max_videos: int = 3) -> Dict:
        """Run the complete pipeline with the given search query.
        
        Args:
            search_query (str): The search query for YouTube videos
            max_videos (int): Maximum number of videos to process
            
        Returns:
            Dict: Complete pipeline results including all steps
        """
        pipeline_start_time = time.time()
        
        print("=" * 80)
        print(f"🎬 YOUTUBE PROCESSING PIPELINE")
        print(f"📝 Search Query: '{search_query}'")
        print(f"🎯 Max Videos: {max_videos}")
        print(f"⏰ Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Step 1: Search for videos
        print("\n" + "=" * 50)
        print("📺 STEP 1: VIDEO SEARCH")
        print("=" * 50)
        
        if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
            print("❌ YouTube API key not set. Please set YOUTUBE_API_KEY.")
            return {"success": False, "error": "YouTube API key not set"}
        
        videos = search_youtube_videos(search_query, max_videos)
        
        if not videos:
            print("❌ No videos found. Pipeline terminated.")
            return {
                "success": False,
                "error": "No videos found for the search query",
                "search_query": search_query,
                "videos_found": 0
            }
        
        # Save search metadata
        search_metadata = {
            "search_query": search_query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_videos_found": len(videos),
            "videos": videos
        }
        
        search_metadata_path = os.path.join(
            self.metadata_folder, f"search_results_{int(time.time())}.json"
        )
        with open(search_metadata_path, "w", encoding="utf-8") as f:
            json.dump(search_metadata, f, indent=2, ensure_ascii=False)
        
        # Step 2: Fetch transcripts
        print("\n" + "=" * 50)
        print("📥 STEP 2: TRANSCRIPT FETCHING")
        print("=" * 50)
        
        # Extract URLs from video data
        urls = [video["url"] for video in videos]
        
        # Fetch transcripts
        fetch_results = self.transcript_fetcher.fetch_transcripts(urls)
        
        # Determine successful transcript files
        video_ids = [video["video_id"] for video in videos]
        transcript_paths = self.transcript_fetcher.get_transcript_files(video_ids)
        
        if not transcript_paths:
            print("❌ No transcripts could be fetched. Pipeline terminated.")
            return {
                "success": False,
                "error": "No transcripts could be fetched",
                "search_query": search_query,
                "videos_found": len(videos),
                "transcripts_fetched": 0
            }
        
        # Step 3: Summarize transcripts
        print("\n" + "=" * 50)
        print("🤖 STEP 3: TRANSCRIPT SUMMARIZATION")
        print("=" * 50)
        
        if not self.summarizer:
            print("⚠️  OpenRouter API key not set. Skipping summarization.")
            summarization_results = {}
        else:
            summarization_results = self.summarizer.summarize_transcripts(
                transcript_paths, videos, self.summaries_folder
            )
        
        # Calculate final results
        pipeline_end_time = time.time()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        successful_summaries = sum(summarization_results.values()) if summarization_results else 0
        
        # Create final results
        final_results = {
            "success": True,
            "search_query": search_query,
            "pipeline_duration_seconds": round(pipeline_duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos_found": len(videos),
            "transcripts_fetched": len(transcript_paths),
            "summaries_created": successful_summaries,
            "output_folder": self.output_folder,
            "videos": videos,
            "transcript_paths": transcript_paths,
            "summarization_results": summarization_results
        }
        
        # Save final results
        results_path = os.path.join(
            self.metadata_folder, f"pipeline_results_{int(time.time())}.json"
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED")
        print("=" * 80)
        print(f"📝 Search Query: '{search_query}'")
        print(f"📺 Videos Found: {len(videos)}")
        print(f"📄 Transcripts Fetched: {len(transcript_paths)}")
        print(f"🤖 Summaries Created: {successful_summaries}")
        print(f"⏱️  Total Duration: {pipeline_duration:.2f} seconds")
        print(f"📁 Output Folder: {self.output_folder}")
        print(f"💾 Results Saved: {results_path}")
        print("=" * 80)
        
        return final_results

print("✅ Pipeline class defined and ready to use!")

# ======================================================================
# ## 6. Run the Complete Pipeline
#
# Execute the complete pipeline with a sample query and display results.
# ======================================================================

# Initialize the pipeline
pipeline = YouTubePipeline(output_folder=CONFIG["output"]["folder"])

# Define our search query
search_query = "LlamaIndex tutorial"
max_videos = 3 

print(f"🚀 Ready to run pipeline with query: '{search_query}'")
print(f"🎯 Max videos: {max_videos}")

# Check if API keys are set
if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
    print("\n⚠️  To run the pipeline, please set your API keys:")
    print("1. YOUTUBE_API_KEY = 'your_youtube_api_key'")
    print("2. OPENROUTER_API_KEY = 'your_openrouter_api_key'")
    print("\nThen re-run this cell to execute the pipeline.")
else:
    print("\n🎬 Starting pipeline...")
    # Run the complete pipeline
    results = pipeline.run_pipeline(search_query, max_videos)
    
    # Display summary of results
    if results["success"]:
        print(f"\n📊 FINAL SUMMARY:")
        print(f"✅ Successfully processed {results['videos_found']} videos")
        print(f"📄 Fetched {results['transcripts_fetched']} transcripts")
        print(f"🤖 Generated {results['summaries_created']} summaries")
        print(f"⏱️  Completed in {results['pipeline_duration_seconds']} seconds")
        print(f"\n📁 Check your results in: {results['output_folder']}")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")

# ======================================================================
# ## 🎉 Tutorial Complete!
#
# You've successfully built a complete YouTube video processing pipeline! 
#
# ### 🛠️ What You Built
#
# 1. **🔍 Video Search** - YouTube Data API integration with duration parsing
# 2. **📥 Transcript Extraction** - yt-dlp integration with error handling
# 3. **🤖 AI Summarization** - OpenRouter integration with structured prompts
# 4. **🔗 Complete Pipeline** - Automated end-to-end processing
#
# ### 📊 Pipeline Architecture
#
# ```
# Search Query → YouTube API → Video URLs → Transcript Fetcher → AI Summarizer → Results
# ```
#
# ### 🚀 Next Steps
#
# **Immediate Improvements:**
# - Modify the search query to explore different topics
# - Adjust `max_videos` parameter for batch processing
# - Customize the AI prompt for specific use cases
#
# **Advanced Features:**
# - Add video filtering (duration, views, upload date)
# - Implement parallel processing for multiple videos
# - Create a web interface for easier usage
# - Add database storage for results
#
# ### 🔑 API Setup
#
# **YouTube Data API (Free):**
# - Visit [Google Cloud Console](https://console.cloud.google.com/)
# - Enable YouTube Data API v3
# - Create and copy API key
#
# **OpenRouter (Affordable AI):**
# - Visit [OpenRouter](https://openrouter.ai/keys)
# - Sign up and generate API key
# - Add credits (starts from $5)
#
# ### 📁 Output Files
#
# Your pipeline generates:
# - `transcripts/*.srt` - Original video transcripts
# - `summaries/*.json` - Structured AI summaries
# - `metadata/*.json` - Search and pipeline metadata
#
# Happy coding! 🚀
# ======================================================================
