import os
from typing import Optional
from serpapi import SerpApiClient
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()


# (Your Pydantic models FlightsInput and FlightsInputSchema remain the same)
class FlightsInput(BaseModel):
    departure_airport: Optional[str] = Field(description='Departure airport code (IATA)')
    arrival_airport: Optional[str] = Field(description='Arrival airport code (IATA)')
    outbound_date: Optional[str] = Field(description='Parameter defines the outbound date. The format is YYYY-MM-DD. e.g. 2024-06-22')
    return_date: Optional[str] = Field(description='Parameter defines the return date. The format is YYYY-MM-DD. e.g. 2024-06-28')
    adults: Optional[int] = Field(1, description='Parameter defines the number of adults. Default to 1.')
    children: Optional[int] = Field(0, description='Parameter defines the number of children. Default to 0.')
    infants_in_seat: Optional[int] = Field(0, description='Parameter defines the number of infants in seat. Default to 0.')
    infants_on_lap: Optional[int] = Field(0, description='Parameter defines the number of infants on lap. Default to 0.')

class FlightsInputSchema(BaseModel):
    params: FlightsInput


@tool(args_schema=FlightsInputSchema)
def flights_finder(params: FlightsInput):
    '''
    Find flights using the Google Flights engine.

    Returns:
        dict: Flight search results.
    '''

    api_params = {
        'api_key': os.getenv('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'hl': 'en',
        'gl': 'us',
        'departure_id': params.departure_airport,
        'arrival_id': params.arrival_airport,
        'outbound_date': params.outbound_date,
        'return_date': params.return_date,
        'currency': 'USD',
        'adults': params.adults,
        'infants_in_seat': params.infants_in_seat,
        'stops': '1',
        'infants_on_lap': params.infants_on_lap,
        'children': params.children
    }

    try:
        client = SerpApiClient(api_params)
        search_results = client.get_dict()

        # --- NEW: Check for an explicit error message from the API ---
        if 'error' in search_results:
            return f"An API error occurred: {search_results['error']}"

        # --- IMPROVED: Check for the key before trying to access it ---
        if 'best_flights' in search_results:
            return search_results['best_flights']
        else:
            # Provide a clear, helpful message if no flights are found
            return "No flights were found for the specified criteria. Please try different dates or airports."

    except Exception as e:
        # Fallback for other unexpected errors
        return f"An unexpected error occurred: {str(e)}"