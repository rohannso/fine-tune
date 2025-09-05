import os
from typing import Optional

# --- Import the specific client from the serpapi library ---
from serpapi import SerpApiClient
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

class HotelsInput(BaseModel):
    q: str = Field(description='Location of the hotel (e.g., "Miami Beach, Florida")')
    check_in_date: str = Field(description='Check-in date. The format is YYYY-MM-DD. e.g. 2025-11-20')
    check_out_date: str = Field(description='Check-out date. The format is YYYY-MM-DD. e.g. 2025-11-25')
    # --- Changed default to a string for robustness ---
    sort_by: Optional[str] = Field("8", description='Parameter for sorting results. Default is "8" for highest rating.')
    adults: Optional[int] = Field(1, description='Number of adults. Default to 1.')
    children: Optional[int] = Field(0, description='Number of children. Default to 0.')
    rooms: Optional[int] = Field(1, description='Number of rooms. Default to 1.')
    hotel_class: Optional[str] = Field(
        None, description='Filter by hotel class (e.g., "2,3,4" for 2, 3, and 4-star hotels).')

class HotelsInputSchema(BaseModel):
    params: HotelsInput

@tool(args_schema=HotelsInputSchema)
def hotels_finder(params: HotelsInput):
    '''
    Find hotels using the Google Hotels engine.

    Returns:
        dict or str: Hotel search results or an error message.
    '''
    # --- Renamed to api_params to avoid confusion with the input 'params' object ---
    api_params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'in',
        'q': params.q,
        'check_in_date': params.check_in_date,
        'check_out_date': params.check_out_date,
        'currency': 'INR',
        'adults': params.adults,
        'children': params.children,
        'rooms': params.rooms,
        'sort_by': params.sort_by,
        'hotel_class': params.hotel_class
    }

    try:
        # --- Use the modern, class-based method for the API call ---
        client = SerpApiClient(api_params)
        search_results = client.get_dict()

        # --- 1. Check for an explicit error message from the API ---
        if 'error' in search_results:
            return f"An API error occurred: {search_results['error']}"

        # --- 2. Check for the 'properties' key before trying to access it ---
        if 'properties' in search_results and search_results['properties']:
            # Return the top 5 results if they exist
            return search_results['properties'][:5]
        else:
            # --- 3. Provide a clear, helpful message if no hotels are found ---
            return "No hotels were found for the specified criteria. Please try different dates or locations."

    except Exception as e:
        # --- 4. Fallback for other unexpected errors (e.g., network issues) ---
        return f"An unexpected error occurred: {str(e)}"
