#!/usr/bin/env python3
"""
Test LLM connectivity for the Vision-Language Embodied Agent.

Usage:
    python test_llm.py
"""

import os
import sys

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠ python-dotenv not installed, using system env vars")

def test_env_vars():
    """Test environment variables."""
    print("\n=== Environment Variables ===")

    api_key = os.environ.get('OPENAI_API_KEY', '')
    api_base = os.environ.get('OPENAI_API_BASE', '')

    print(f"OPENAI_API_KEY: {'✓ Set' if api_key else '✗ Not set'}")
    if api_key:
        print(f"  Value: {api_key[:20]}...{api_key[-4:]}")

    print(f"OPENAI_API_BASE: {'✓ Set' if api_base else '○ Not set (using default)'}")
    if api_base:
        print(f"  Value: {api_base}")

    return api_key, api_base

def test_langchain(api_key: str, api_base: str):
    """Test LangChain LLM connection."""
    print("\n=== LangChain LLM Test ===")

    try:
        from langchain_openai import ChatOpenAI
        print("✓ langchain_openai imported")
    except ImportError as e:
        print(f"✗ Failed to import langchain_openai: {e}")
        return False

    # Test with different models
    models_to_test = ["gpt-5.2", "gpt-4o-mini", "gpt-4o", "gpt-4"]

    for model in models_to_test:
        print(f"\nTesting model: {model}")

        try:
            kwargs = {
                "model": model,
                "temperature": 0.1,
                "api_key": api_key,
                "max_tokens": 100
            }

            if api_base:
                kwargs["base_url"] = api_base

            llm = ChatOpenAI(**kwargs)
            print(f"  ✓ LLM initialized")

            # Test simple call
            response = llm.invoke("Say 'hello' in one word.")
            print(f"  ✓ Response: {response.content[:50]}...")
            return True, model

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return False, None

def test_task_decomposer(api_key: str, api_base: str):
    """Test TaskDecomposer."""
    print("\n=== TaskDecomposer Test ===")

    try:
        from src.planning.task_decomposer import TaskDecomposer
        print("✓ TaskDecomposer imported")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    decomposer = TaskDecomposer(use_llm=True, api_key=api_key)
    print(f"LLM initialized: {'✓' if decomposer._llm else '✗'}")

    if not decomposer._llm:
        print("Cannot test decomposition without LLM")
        return False

    # Test decomposition
    test_tasks = [
        "找到苹果",
        "捡起苹果",
        "走到门口",
        "打开冰箱",
    ]

    for task in test_tasks:
        print(f"\nTask: {task}")
        result = decomposer.decompose(task)
        print(f"  Subgoals: {len(result.subgoals)}")
        for sg in result.subgoals:
            print(f"    - {sg.action}: {sg.target}")

    return True

def main():
    print("=" * 50)
    print("LLM Connectivity Test")
    print("=" * 50)

    # Test env vars
    api_key, api_base = test_env_vars()

    if not api_key:
        print("\n✗ No API key found. Please set OPENAI_API_KEY in .env file")
        return 1

    # Test LangChain
    success, working_model = test_langchain(api_key, api_base)

    if not success:
        print("\n✗ No working LLM model found")
        return 1

    print(f"\n✓ Working model: {working_model}")

    # Test TaskDecomposer
    test_task_decomposer(api_key, api_base)

    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)

    return 0

if __name__ == "__main__":
    sys.exit(main())
