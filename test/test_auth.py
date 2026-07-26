#!/usr/bin/env python
"""
Test script for InternTA API with bearer token authentication
"""

import os
import pytest
import requests
from dotenv import load_dotenv

# Load the API token from .env file
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN", "")

# Base URL for the API
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:6006")


@pytest.fixture
def api_url():
    """Fixture for the API URL"""
    return f"{BASE_URL}/v1/chat/completions"


@pytest.fixture
def test_payload():
    """Fixture for the test payload"""
    return {
        "model": "internTA",
        "messages": [{"role": "user", "content": "Hello, what can you help me with?"}]
    }


@pytest.fixture
def auth_headers():
    """Fixture for headers with authentication"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }


@pytest.fixture
def no_auth_headers():
    """Fixture for headers without authentication"""
    return {
        "Content-Type": "application/json"
    }


def test_with_valid_authentication(api_url, test_payload, auth_headers):
    """Test the /v1/chat/completions endpoint with authentication"""
    
    # Make the request
    response = requests.post(api_url, json=test_payload, headers=auth_headers)
    
    # Check the response
    assert response.status_code == 200, f"Authentication failed with status code: {response.status_code}, response: {response.text}"
    
    # Additional assertions to verify response format
    response_json = response.json()
    assert "choices" in response_json, "Response is missing 'choices' field"


def test_without_authentication(api_url, test_payload, no_auth_headers):
    """Test the endpoint without authentication (should fail)"""
    
    # Make the request
    response = requests.post(api_url, json=test_payload, headers=no_auth_headers)
    
    # Check the response - should be 401 Unauthorized
    assert response.status_code == 401, f"Expected 401 status code, got {response.status_code}. Response: {response.text}" 