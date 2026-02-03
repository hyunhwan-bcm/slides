def get_weather(city: str, units: str = "fahrenheit") -> dict:
    """Mock weather API call."""
    weather_data = {
        "New York": {"temp": 72, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Rainy"},
        "Tokyo": {"temp": 28, "condition": "Cloudy"}
    }
    
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    
    if units == "celsius":
        data["temp"] = int((data["temp"] - 32) * 5/9)
    
    return {
        "city": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "units": units
    }
