"""Tools for generating natural-language descriptions of PNG images."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image, UnidentifiedImageError


class ImageDescriber:
    """Generate short captions for PNG files using Azure OpenAI when available."""

    def __init__(
        self,
        *,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self._api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        self._deployment_name = deployment_name or os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT_NAME") or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        )

        self._client: Optional[AzureOpenAI]
        if all([self._azure_endpoint, self._api_key, self._deployment_name]):
            self._client = AzureOpenAI(
                azure_endpoint=self._azure_endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        else:
            self._client = None

    def describe(self, image_path: str, *, prompt: Optional[str] = None) -> str:
        """
        Attempt to produce a natural-language description for a PNG file.

        If Azure OpenAI credentials are configured, the image is sent to the Responses API.
        Otherwise (or if the API call fails) we return a lightweight description with basic metadata.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
                mode = img.mode
        except UnidentifiedImageError as exc:
            raise ValueError(f"Unable to open image at {image_path}") from exc

        if not path.suffix.lower().endswith("png"):
            raise ValueError(f"Expected a PNG image, received: {path.suffix or 'unknown extension'}")

        base_prompt = prompt or (
            "You are an observant visual assistant. Describe the main elements, colors, and context "
            "of the provided image in two or three concise sentences."
        )

        if self._client:
            image_bytes = path.read_bytes()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            try:
                response = self._client.responses.create(
                    model=self._deployment_name,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": base_prompt}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Provide the description now."},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{encoded_image}"},
                            ],
                        },
                    ],
                )
                if hasattr(response, "output_text"):
                    description = response.output_text
                else:
                    output = getattr(response, "output", None)
                    description = ""
                    if isinstance(output, list):
                        for item in output:
                            content = item.get("content") if isinstance(item, dict) else None
                            if isinstance(content, list):
                                for detail in content:
                                    if detail.get("type") == "output_text":
                                        description += detail.get("text", "")
                    description = description.strip()

                if description:
                    return description.strip()
            except Exception as e:
                print(f"Failed to generate description: {str(e)}")
                # Fall back to heuristic description if the API call fails.
                pass

        width_height_description = f"The image is {width} by {height} pixels with {mode} color mode."
        return (
            f"{width_height_description} A detailed description could not be generated automatically. "
            "Please try again once the Azure OpenAI configuration is available."
        )


def describe_image(image_path: str, *, prompt: Optional[str] = None) -> str:
    """Convenience wrapper that uses default environment-backed configuration."""
    return ImageDescriber().describe(image_path, prompt=prompt)
