# -*- coding: utf-8 -*-
"""
Quick Test Script - Verify Module Setup
========================================
"""

import sys
import os

print("Testing Module 6 Setup...")
print("="*60)

# Test 1: Check if we can import required packages
print("\n1. Testing imports...")
try:
    from langchain_openai import ChatOpenAI
    print("   ✓ langchain_openai")
except ImportError as e:
    print(f"   ✗ langchain_openai: {e}")

try:
    from langchain.agents import initialize_agent, AgentExecutor, AgentType
    print("   ✓ langchain.agents")
except ImportError as e:
    print(f"   ✗ langchain.agents: {e}")

try:
    from langchain_community.vectorstores import Chroma
    print("   ✓ langchain_community.vectorstores")
except ImportError as e:
    print(f"   ✗ langchain_community.vectorstores: {e}")

try:
    from langchain_core.tools import Tool
    print("   ✓ langchain_core.tools")
except ImportError as e:
    print(f"   ✗ langchain_core.tools: {e}")

# Test 2: Check if data file exists
print("\n2. Testing data file access...")
data_path = '../../data/synthetic_tickets.json'
if os.path.exists(data_path):
    print(f"   ✓ Found {data_path}")
    import json
    with open(data_path, 'r') as f:
        tickets = json.load(f)
    print(f"   ✓ Loaded {len(tickets)} tickets")
else:
    print(f"   ✗ Cannot find {data_path}")

# Test 3: Check environment variables
print("\n3. Testing environment variables...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"   ✓ OPENAI_API_KEY is set ({api_key[:8]}...)")
else:
    print("   ✗ OPENAI_API_KEY not found")

# Test 4: Try importing our tools
print("\n4. Testing custom tools module...")
try:
    from tools import SupportTicketTools
    print("   ✓ Successfully imported SupportTicketTools")
    
    # Try initializing
    tool_manager = SupportTicketTools()
    tools = tool_manager.get_tools()
    print(f"   ✓ Created {len(tools)} tools:")
    for tool in tools:
        print(f"      - {tool.name}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Setup test complete!")
print("\nIf all checks passed, run: python demo.py")
