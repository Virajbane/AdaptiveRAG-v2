"""
Weather tool - Get current weather for a location using OpenWeatherMap API

2026-08-09: Created to support weather queries in RAG system
"""

from typing import Dict
from app.config.settings import settings


class WeatherTool:
    """
    OpenWeatherMap weather tool.
    Fetches current weather data for a given location.
    """

    def __init__(self):
        self.api_key = settings.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    async def get_weather(self, location: str) -> Dict:
        """
        Get current weather for a location.

        Args:
            location: City name (e.g., 'London', 'Mumbai')

        Returns:
            {
                "location": str,
                "temperature": float,
                "description": str,
                "humidity": int,
                "wind_speed": float,
                "feels_like": float,
                "pressure": int,
                "error": str (if failed)
            }
        """
        try:
            if not self.api_key:
                return {"error": "OpenWeatherMap API key not configured"}

            import aiohttp

            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"  # Get temperature in Celsius
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        data = await response.json()
                        error_msg = data.get('message', 'Unknown error')
                        return {"error": f"Weather API error: {error_msg}"}

                    data = await response.json()
                    
                    # Extract weather information
                    return {
                        "location": data.get("name"),
                        "country": data.get("sys", {}).get("country"),
                        "temperature": data["main"].get("temp"),
                        "feels_like": data["main"].get("feels_like"),
                        "temp_min": data["main"].get("temp_min"),
                        "temp_max": data["main"].get("temp_max"),
                        "pressure": data["main"].get("pressure"),
                        "humidity": data["main"].get("humidity"),
                        "wind_speed": data["wind"].get("speed"),
                        "wind_degree": data["wind"].get("deg"),
                        "cloudiness": data.get("clouds", {}).get("all"),
                        "description": data["weather"][0].get("description"),
                        "main": data["weather"][0].get("main"),
                        "sunrise": data.get("sys", {}).get("sunrise"),
                        "sunset": data.get("sys", {}).get("sunset"),
                    }

        except Exception as e:
            return {"error": f"Weather API call failed: {str(e)}"}


# Global weather tool instance
weather_tool = WeatherTool()