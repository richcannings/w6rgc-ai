#!/usr/bin/env python3
# arrl_news_helper.py - ARRL News Utility
#
# This module provides functions for fetching and summarizing ARRL news
# from the ARNewsline website. It is used by the W6RGC/AI Ham
# Radio Voice Assistant to provide news updates via voice commands.
#
# Key Features:
#  - Fetch latest ARRL news from arnewsline.org
#  - Parse HTML to extract news headlines and content
#  - Format news for natural language presentation
#  - Cache results to avoid excessive requests
#
# Usage:
#  from arrl_news_helper import get_arrl_news_summary
#  news_summary = get_arrl_news_summary()
#  print(news_summary)
#
# Author: Rich Cannings <rcannings@gmail.com>
# Copyright 2025 Rich Cannings
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional

# ARNewsline website URL
ARRL_NEWS_URL = "https://www.arnewsline.org/"

# Cache settings
NEWS_CACHE_DURATION = 300  # 5 minutes
_last_fetch_time = 0
_cached_news = None

def get_arrl_news_summary(max_headlines: int = 5) -> str:
    """
    Fetch and summarize ONLY the latest ARRL news report from ARNewsline.
    
    Args:
        max_headlines (int): (Ignored) for compatibility, always returns only the latest report
        
    Returns:
        str: Natural language summary of the latest ARRL news report
    """
    global _last_fetch_time, _cached_news
    
    # Check if we have recent cached data
    current_time = time.time()
    if _cached_news and (current_time - _last_fetch_time) < NEWS_CACHE_DURATION:
        print("📰 Using cached ARRL news data")
        return _format_news_summary(_cached_news, 1)
    
    try:
        print("📰 Fetching latest ARRL news from ARNewsline...")
        response = requests.get(ARRL_NEWS_URL, timeout=10)
        response.raise_for_status()
        
        # Parse the news from the HTML
        news_items = _parse_arrl_news(response.text)
        
        # Cache the results
        _cached_news = news_items
        _last_fetch_time = current_time
        
        return _format_news_summary(news_items, 1)
        
    except requests.RequestException as e:
        print(f"❌ Error fetching ARRL news: {e}")
        return "I'm sorry, I was unable to fetch the latest ARRL news at this time. Please try again later."
    except Exception as e:
        print(f"❌ Error processing ARRL news: {e}")
        return "I encountered an error while processing the ARRL news. Please try again later."

def _parse_arrl_news(html_content: str) -> List[Dict[str, str]]:
    """
    Parse HTML content to extract ARRL news items.
    
    Args:
        html_content (str): HTML content from ARNewsline website
        
    Returns:
        List[Dict[str, str]]: List of news items with title, date, and content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    news_items = []
    
    # Look for article elements that contain news reports
    articles = soup.find_all('article', class_='hentry')
    
    for article in articles:
        # Find the h1 title
        title_elem = article.find('h1', class_='entry-title')
        if not title_elem:
            continue
            
        title_link = title_elem.find('a')
        if not title_link:
            continue
            
        title_text = title_link.get_text(strip=True)
        
        # Look for patterns like "Amateur Radio Newsline Report 2486 for Friday, June 20th, 2025"
        if "Report" in title_text and any(month in title_text for month in ["January", "February", "March", "April", "May", "June", 
                                                               "July", "August", "September", "October", "November", "December"]):
            
            # Extract the report number and date
            parts = title_text.split()
            report_number = None
            date_info = ""
            
            for i, part in enumerate(parts):
                if part == "Report" and i + 1 < len(parts):
                    try:
                        report_number = int(parts[i + 1])
                    except ValueError:
                        pass
                if any(month in part for month in ["January", "February", "March", "April", "May", "June", 
                                                  "July", "August", "September", "October", "November", "December"]):
                    date_info = " ".join(parts[i:])
                    break
            
            # Look for the content in the body
            body_elem = article.find('div', class_='body')
            if body_elem:
                # Find the paragraph with headlines (has white-space:pre-wrap style)
                headline_para = body_elem.find('p', style=lambda x: x and 'white-space:pre-wrap' in x)
                if headline_para:
                    # Get text content and replace <br> tags with newlines
                    content = str(headline_para)
                    # Replace <br> tags with newlines
                    content = content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
                    # Parse again to get clean text
                    clean_soup = BeautifulSoup(content, 'html.parser')
                    content = clean_soup.get_text(strip=True)
                else:
                    content = ""
            else:
                content = ""
            
            if report_number and date_info:
                news_items.append({
                    "report_number": report_number,
                    "date": date_info,
                    "title": title_text,
                    "content": content
                })
    
    # Sort by report number (newest first)
    news_items.sort(key=lambda x: x["report_number"], reverse=True)
    
    return news_items

def _format_news_summary(news_items: List[Dict[str, str]], max_headlines: int) -> str:
    """
    Format news items into a natural language summary.
    
    Args:
        news_items (List[Dict[str, str]]): List of parsed news items
        max_headlines (int): Maximum number of headlines to include
        
    Returns:
        str: Formatted news summary for TTS
    """
    if not news_items:
        return "No recent ARRL news was found."
    
    # Take the most recent news items
    recent_news = news_items[:max_headlines]
    
    summary = f"Here are the {len(recent_news)} most recent ARRL news headlines from Amateur Radio Newsline.\n\n"
    
    for i, news in enumerate(recent_news, 1):
        # Extract key headlines from the content
        headlines = _extract_headlines(news.get("content", ""))
        
        summary += f"Report {news['report_number']}, dated {news['date']}.\n"
        
        if headlines:
            summary += f"The main stories include: {headlines}\n"
        else:
            summary += "News content is available but could not be parsed.\n"
        
        summary += "\n"
    
    summary += "End of ARRL news summary."
    return summary

def _extract_headlines(content: str) -> str:
    """
    Extract key headlines from news content.
    
    Args:
        content (str): Raw news content
        
    Returns:
        str: Extracted headlines formatted for speech
    """
    if not content:
        return ""
    
    # Split content into lines and look for headlines that start with " - "
    lines = content.split('\n')
    headlines = []
    
    for line in lines:
        line = line.strip()
        # Look for lines that start with " - " (dash space)
        if line.startswith(' - '):
            # Remove the " - " prefix and clean up
            headline = line[3:].strip()
            if headline and len(headline) > 5:  # Only include substantial headlines
                headlines.append(headline)
    
    # If we found headlines, return them
    if headlines:
        # Limit to 5 headlines for brevity and join with periods
        return ". ".join(headlines[:5])
    
    # If no bullet points, try to extract the first few sentences
    sentences = content.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 20:
            return first_sentence
    
    return "News content available but format not recognized."

def clear_news_cache():
    """Clear the cached news data to force a fresh fetch."""
    global _last_fetch_time, _cached_news
    _last_fetch_time = 0
    _cached_news = None
    print("📰 ARRL news cache cleared") 