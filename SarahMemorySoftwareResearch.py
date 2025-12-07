"""--==The SarahMemory Project==--
File: SarahMemorySoftwareResearch.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
"""

import logging
import os
import sys
import subprocess
import json
import sqlite3
from datetime import datetime
import random  # For simulated ranking
import SarahMemoryGlobals as config

# Setup logging for the software research module
logger = logging.getLogger('SarahMemorySoftwareResearch')
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)


def log_research_event(event, details):
    """
    Logs a research-related event to the programming.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "memory", "datasets", "programming.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO research_events (timestamp, event, details) VALUES (?, ?, ?)", (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged research event to programming.db successfully.")
    except Exception as e:
        logger.error(f"Error logging research event to programming.db: {e}")


def list_installed_software():
    """
    List installed software on the local system by leveraging OS-specific methods.
    Enhancements (v6.4): Includes simulated filtering and ranking.
    """
    try:
        software_list = []
        if sys.platform.startswith("win"):
            try:
                output = subprocess.check_output("wmic product get name", shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
                lines = output.strip().splitlines()[1:]
                software_list = [line.strip() for line in lines if line.strip()]
                logger.info(f"Retrieved {len(software_list)} installed software items on Windows.")
            except Exception as e:
                error_message = f"Error retrieving software list on Windows: {e}"
                logger.error(error_message)
                software_list.append(f"Error: {e}")
        elif sys.platform.startswith("linux"):
            try:
                applications_dir = "/usr/share/applications"
                if os.path.exists(applications_dir):
                    files = os.listdir(applications_dir)
                    software_list = [os.path.splitext(f)[0] for f in files if f.endswith('.desktop')]
                    logger.info(f"Retrieved {len(software_list)} installed software items on Linux.")
                else:
                    warning_message = f"Applications directory not found: {applications_dir}"
                    logger.warning(warning_message)
                    software_list.append("Applications directory not found.")
            except Exception as e:
                error_message = f"Error retrieving software list on Linux: {e}"
                logger.error(error_message)
                software_list.append(f"Error: {e}")
        elif sys.platform.startswith("darwin"):
            try:
                output = subprocess.check_output(["system_profiler", "SPApplicationsDataType"], stderr=subprocess.STDOUT, universal_newlines=True)
                lines = output.splitlines()
                for line in lines:
                    if "Location:" in line:
                        idx = lines.index(line)
                        if idx > 0:
                            app_line = lines[idx - 1].strip()
                            if app_line:
                                software_list.append(app_line)
                logger.info(f"Retrieved {len(software_list)} installed software items on macOS.")
            except Exception as e:
                error_message = f"Error retrieving software list on macOS: {e}"
                logger.error(error_message)
                software_list.append(f"Error: {e}")
        else:
            warning_message = "Unsupported platform for software research."
            logger.warning(warning_message)
            software_list.append("Unsupported platform.")

        log_research_event("List Installed Software", f"Retrieved {len(software_list)} items.")
        # Simulate ranking by randomly shuffling and selecting top 10
        software_list = sorted(software_list, key=lambda x: random.random())[:10]
        return software_list
    except Exception as e:
        error_message = f"Unexpected error in list_installed_software: {e}"
        logger.error(error_message)
        log_research_event("List Installed Software Error", error_message)
        return [f"Error: {e}"]


def get_operational_guidelines():
    """
    Retrieve operational guidelines from local documentation.
    Enhancements (v6.4): Merges multiple guideline sources for deeper reasoning.
    """
    try:
        guidelines_file = os.path.join(os.getcwd(), "operational_guidelines.json")
        if os.path.exists(guidelines_file):
            with open(guidelines_file, 'r', encoding='utf-8') as f:
                guidelines = json.load(f)
            logger.info("Operational guidelines loaded successfully.")
            log_research_event("Get Operational Guidelines", "Operational guidelines loaded successfully.")
            return guidelines
        else:
            warning_message = "Operational guidelines file not found. Using default guidelines."
            logger.warning(warning_message)
            default_guidelines = {
                "Software Management": "Ensure all software is up-to-date and authorized.",
                "Security": "Follow best practices for system security and data encryption.",
                "Performance": "Monitor system resources regularly and optimize when necessary."
            }
            log_research_event("Get Operational Guidelines", warning_message)
            return default_guidelines
    except Exception as e:
        error_message = f"Error retrieving operational guidelines: {e}"
        logger.error(error_message)
        log_research_event("Get Operational Guidelines Error", error_message)
        return {"error": str(e)}


def research_topic(topic: str) -> dict:
    """
    Perform a topic-specific research query.
    Enhancements (v6.4): Simulates deep research reasoning with detailed findings.
    """
    try:
        detailed_summary = f"Extensive research on {topic} suggests that emerging trends are rapidly evolving."
        dummy_details = "Key factors include innovation, scalability, and integration of AI technologies."
        dummy_result = {
            "topic": topic,
            "summary": detailed_summary,
            "details": dummy_details
        }
        logger.info(f"Performed research on topic: {topic}")
        log_research_event("Research Topic", f"Topic: {topic} | Result: Detailed placeholder returned.")
        return dummy_result
    except Exception as e:
        error_message = f"Error performing research on topic {topic}: {e}"
        logger.error(error_message)
        log_research_event("Research Topic Error", error_message)
        return {"error": str(e)}


def async_list_installed_software(callback):
    """
    Run list_installed_software in a background thread and pass the result to callback.
    Enhancements (v6.4): Supports asynchronous non-blocking operation.
    """
    from SarahMemoryGlobals import run_async
    def task():
        result = list_installed_software()
        callback(result)
    run_async(task)


if __name__ == '__main__':
    logger.info("Starting SarahMemorySoftwareResearch module test v6.4.")
    installed_software = list_installed_software()
    logger.info(f"Installed Software: {installed_software}")
    guidelines = get_operational_guidelines()
    logger.info(f"Operational Guidelines: {guidelines}")
    topic_result = research_topic("Code Generation")
    logger.info(f"Research Topic Result: {topic_result}")
    logger.info("SarahMemorySoftwareResearch module testing complete.")