import uuid
from pathlib import Path
import requests
from decouple import config
from langchain.tools import tool
from openai import OpenAI
from pydantic import BaseModel, Field

IMAGE_DIRECTORY = Path(__file__).parent.parent / "images"
CLIENT = OpenAI(api_key=str(config("OPENAI_API_KEY")))

# create a helper function that takes an image URL and downloads and saves that image to our /images folder. 
def image_downloader(image_url: str | None) -> str:
    if image_url is None:
        return "No image URL returned from API."

    response = requests.get(image_url)
    if response.status_code != 200:
        return f"Failed to download image from {image_url}."
    
    unique_id: uuid.UUID = uuid.uuid4()
    image_path = IMAGE_DIRECTORY / f"{unique_id}.png"

    with open(image_path, "wb") as file:
        file.write(response.content)

    return str(image_path)

class GenerateImageInput(BaseModel):
  image_description: str = Field( 
    description="A detailed description of the desired image."
)
  
@tool("generate-image", args_schema=GenerateImageInput)
def generate_image(image_description: str) -> str:
    """Generate an image based on a detailed description."""
    response = CLIENT.images.generate(
        model="dall-e-3",
        prompt=image_description,
        size="1024x1024",
        quality="standard",  
        n=1
    )
    image_url = response.data[0].url
    image_path = image_downloader(image_url)
    return image_downloader(image_url)

if __name__ == "__main__":
    print(generate_image.run("A cute cat with wings flying over a rainbow."))