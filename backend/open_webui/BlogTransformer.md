# Google GenAI - Blog Transformer

This is the source code for Google GenAI function for Open WebUI

```python
"""
title: Gemini Manifold Pipe
author: mohit-srivastava
author_url: https://github.com/iammohit1311
funding_url: https://github.com/open-webui
version: 0.1.4
license: MIT
"""

import base64
import os
import json
import re

import requests
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator

# Set DEBUG to True to enable detailed logging
DEBUG = True
ngrok = "https://5d01-2a09-bac0-1000-1cd-00-39-f6.ngrok-free.app"

# Base64 specs
mime_type = None
image_data = None


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.id = "google_genai"
        self.type = "manifold"
        self.name = "Google: "
        self.valves = self.Valves(
            **{
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )
        self.generated_html = None  # Store the generated HTML here
        self.isBase64 = False

    def get_google_models(self):
        if not self.valves.GOOGLE_API_KEY:
            return [
                {
                    "id": "error",
                    "name": "GOOGLE_API_KEY is not set. Please update the API Key in the valves.",
                }
            ]
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            models = genai.list_models()
            print("MODELSSSSSSSS -- ", models)
            return [
                {
                    "id": model.name[7:],  # remove the "models/" part
                    "name": model.display_name,
                }
                for model in models
                if "generateContent" in model.supported_generation_methods
                if model.name.startswith("models/")
            ]
        except Exception as e:
            if DEBUG:
                print(f"Error fetching Google models: {e}")
            return [
                {"id": "error", "name": f"Could not fetch models from Google: {str(e)}"}
            ]

    def pipes(self) -> List[dict]:
        return self.get_google_models()

    def inject_base64_image(self, base64_data: str, mime_type: str = "image/jpeg"):
        print("INJECT")
        if not hasattr(self, "generated_html") or not self.generated_html:
            print("[WARN] generated_html is not set or empty. Skipping injection.")
            return
        print(f"MIME Type: {mime_type}")

        new_src_value = f"data:{mime_type};base64,{base64_data}"

        # Nested function to process the content inside the <main> tag
        def replace_in_main(main_match):
            main_content = main_match.group(0)

            # Step 1: Find the first img tag within the main content
            # This regex finds the entire <img> tag, potentially multi-line
            img_tag_search = re.search(
                r"<img\b[^>]*?>",
                main_content,
                re.IGNORECASE | re.DOTALL,  # DOTALL is crucial for multi-line tags
            )

            if not img_tag_search:
                print("[WARN] No <img> tag found inside <main>.")
                return main_content  # Return original content if no img tag at all

            # Get the matched <img> tag string and its start/end indices within main_content
            img_tag_str = img_tag_search.group(0)
            img_start_in_main, img_end_in_main = img_tag_search.span()

            # Step 2: Find the src attribute *within* the found img tag string
            # This regex specifically targets src="..." or src='...' with an http/https URL
            src_attr_search = re.search(
                r'(\bsrc=["\'])(https?:\/\/[^"\']*?)(["\'])',
                img_tag_str,
                re.IGNORECASE,
                # DOTALL is not needed here as we are searching within a single img tag string
            )

            if not src_attr_search:
                print(
                    "[WARN] <img> tag found inside <main>, but no http/https src attribute to replace."
                )
                return main_content  # Return original content if no relevant src found

            # Get the matched groups for src=" or src=', the URL, and the closing quote
            src_prefix = src_attr_search.group(1)  # e.g., 'src="' or "src='"
            src_suffix = src_attr_search.group(3)  # e.g., '"' or "'"

            # Construct the new src attribute string with the data URL
            # We'll use the original quotes found by the regex
            new_src_attr_str = f"{src_prefix}{new_src_value}{src_suffix}"

            # Replace the old src attribute string with the new one within the img_tag_str
            # using string slicing based on the span of the src attribute match
            src_start_in_img, src_end_in_img = src_attr_search.span()
            updated_img_tag_str = (
                img_tag_str[:src_start_in_img]
                + new_src_attr_str
                + img_tag_str[src_end_in_img:]
            )

            print("[SUCCESS] Replaced src in first relevant <img> tag inside <main>.")

            # Replace the original img tag string in the main_content
            # with the updated img tag string using string slicing
            updated_main_content = (
                main_content[:img_start_in_main]
                + updated_img_tag_str
                + main_content[img_end_in_main:]
            )

            return updated_main_content

        # Apply the image replacement by finding the <main> tag first
        # re.sub with a function calls the function with the match object
        # The function returns the string that replaces the matched part (<main> block)
        self.generated_html = re.sub(
            r"<main\b[^>]*?>.*?</main>",
            replace_in_main,
            self.generated_html,
            flags=re.DOTALL | re.IGNORECASE,
            count=1,  # Optional: Process only the first main tag if multiple exist
        )

        print("Finished injection")

    def create_wordpress_post(
        self, content: str, title: str = "Mohit Draft Post", status: str = "draft"
    ) -> tuple[str, bool]:
        """
        Create a WordPress post using the REST API.

        :param title: Title of the post
        :param content: HTML content of the post provided by user
        :param status: Post status (e.g., 'draft', 'publish')
        :return: Response message and success status
        """
        try:
            print(
                "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM"
            )
            url = f"{ngrok}/wordpress/wp-json/wp/v2/posts"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Basic bW9oaXQuc3JpdjpTWVo4IGVGc2EgUUhPdSA0NnZrIGhOUVUgOUVqSw==",
            }
            payload = {"title": title, "content": content, "status": status}

            response = requests.post(url, headers=headers, data=json.dumps(payload))

            if response.status_code in [200, 201]:
                return (
                    f"Post created successfully with ID: {response.json().get('id')}",
                    True,
                )
            else:
                return f"Failed to create post: {response.text}", False
        except Exception as e:
            return f"Error occurred while creating post: {str(e)}", False

    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            print("BODY:", body)
            print("MODEL ID RECEIVED:", body.get("model", ""))
            print(body.get("custom_model_id"))
            model_id = body["model"]

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]

            model_id = model_id.lstrip(".")

            # if not model_id.startswith("gemini-"):
            #     return f"Error: Invalid model name format: {model_id}"

            messages = body["messages"]
            # print(f"MESSAGES ----- {messages}")

            contents = []
            processed_messages = set()  # Track processed messages

            def process_message_content(content, role):
                parts = []
                if isinstance(content, str):
                    if content.strip():
                        parts.append({"text": content})
                elif isinstance(content, list):
                    for item in content:
                        if item["type"] == "text" and item["text"].strip():
                            parts.append({"text": item["text"]})
                        elif item["type"] == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                try:
                                    print("low")
                                    self.isBase64 = True
                                    header, image_data = image_url.split(",", 1)
                                    mime_type = header.split(";")[0].split(":")[1]
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": mime_type,
                                                "data": image_data,
                                            }
                                        }
                                    )
                                except Exception as e:
                                    print(f"Error processing base64 image: {e}")
                            else:
                                parts.append({"image_url": image_url})
                return parts

            # Process metadata files first
            metadata_files = body.get("metadata", {}).get("files", [])
            if metadata_files:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        parts = process_message_content(msg.get("content"), msg["role"])

                        # Add image data from metadata
                        for file in metadata_files:
                            if file.get("type") == "image":
                                image_url = file["url"]
                                if image_url.startswith("data:image"):
                                    try:
                                        print("low2")
                                        self.isBase64 = True
                                        header, image_data = image_url.split(",", 1)
                                        mime_type = header.split(";")[0].split(":")[1]
                                        parts.append(
                                            {
                                                "inline_data": {
                                                    "mime_type": mime_type,
                                                    "data": image_data,
                                                }
                                            }
                                        )
                                    except Exception as e:
                                        print(
                                            f"[ERROR] Failed to process base64 image: {e}"
                                        )
                                else:
                                    parts.append({"image_url": image_url})

                        if parts:
                            contents.append({"role": msg["role"], "parts": parts})
                            processed_messages.add(id(msg))
                        break

            last = messages[-1]

            # Short-circuit if the *last* message is exactly "deploy"
            if (
                last["role"] == "user"
                and isinstance(last["content"], str)
                and last["content"].strip().lower() == "deploy"
                and body.get("custom_model_id") == "wordpress-drafter"
            ):
                return self.deploy()

            # Normal generation flow
            stream = body.get("stream", False)

            if DEBUG:
                # print("Incoming body:", str(body))
                pass

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            # Process remaining messages
            for message in messages:
                if id(message) in processed_messages:
                    continue

                role = message.get("role")
                if role == "system":
                    continue

                parts = []

                # Handle files in message
                if "files" in message and message["files"]:
                    for file in message["files"]:
                        if file["type"] == "image":
                            image_url = file["url"]
                            if image_url.startswith("data:image"):
                                try:
                                    print("low3")
                                    self.isBase64 = True
                                    header, image_data = image_url.split(",", 1)
                                    mime_type = header.split(";")[0].split(":")[1]
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": mime_type,
                                                "data": image_data,
                                            }
                                        }
                                    )
                                except Exception as e:
                                    print(f"Error processing base64 image: {e}")
                            else:
                                parts.append({"image_url": image_url})

                elif isinstance(message.get("content"), list):
                    print("--------------- HELLLLLLLLLLLLLLLOOOOO ---------")
                    # parts = []
                    for content in message["content"]:
                        if content["type"] == "text":
                            parts.append({"text": content["text"]})
                        elif content["type"] == "image_url":
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                # Handle base64 image data
                                try:
                                    print("low4")
                                    self.isBase64 = True
                                    header, image_data = image_url.split(",", 1)
                                    mime_type = header.split(";")[0].split(":")[1]
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": mime_type,
                                                "data": image_data,
                                            }
                                        }
                                    )
                                    if DEBUG:
                                        print(
                                            "Successfully processed base64 image data"
                                        )
                                except Exception as e:
                                    print(f"Error processing base64 image: {e}")
                                    parts.append(
                                        {
                                            "text": f"Error processing base64 image: {str(e)}"
                                        }
                                    )

                            else:
                                # Handle regular image URL
                                parts.append({"image_url": image_url})
                    # contents.append(
                    #     {"role": message["role"], "parts": parts}
                    # )  # Doubtful

                elif isinstance(message.get("content"), str):
                    content = message.get("content")

                    # if DEBUG:
                    #     print(f"Processing string content: {content}")

                    # Check for base64 image
                    if content.startswith("data:image"):
                        try:
                            print("low5")
                            self.isBase64 = True
                            header, image_data = content.split(",", 1)
                            mime_type = header.split(";")[0].split(":")[1]
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": image_data,
                                    }
                                }
                            )
                        except Exception as e:
                            parts.append(
                                {"text": f"Error processing base64 image: {str(e)}"}
                            )

                    elif content.strip().startswith("http"):
                        try:
                            # if DEBUG:
                            #     print(f"Fetching image from URL: {content}")

                            # Assume it's an image URL and fetch it
                            response = requests.get(content, timeout=10)
                            response.raise_for_status()

                            system_message += f"""
                                context: image_url = {content}
                            """

                            image_data = base64.b64encode(response.content).decode(
                                "utf-8"
                            )

                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",  # Adjust MIME type if needed
                                        "data": image_data,
                                    }
                                }
                            )
                            # if DEBUG:
                            #     print("Image fetched and encoded successfully")

                        except Exception as e:
                            # if DEBUG:
                            #     print(f"Failed to fetch image: {e}")

                            parts.append(
                                {"text": f"Error fetching image: {str(e)}"}
                            )  # Fallback to text

                    # keywords = [
                    #     "deploy",
                    #     "post",
                    #     "deploy on wordpress",
                    #     "publish draft",
                    # ]

                    # if any(word in content.lower() for word in keywords):
                    #     self.deploy()

                    else:
                        parts.append({"text": message["content"]})

                    # Handle text content
                    if "content" in message and message["content"]:
                        if (
                            isinstance(message["content"], str)
                            and message["content"].strip()
                        ):
                            parts.insert(0, {"text": message["content"]})
                        elif isinstance(message["content"], list):
                            for content in message["content"]:
                                if content["type"] == "text":
                                    parts.append({"text": content["text"]})
                                elif content["type"] == "image_url":
                                    image_url = content["image_url"]["url"]
                                    if image_url.startswith("data:image"):
                                        try:
                                            print("low6")
                                            self.isBase64 = True
                                            header, image_data = image_url.split(",", 1)
                                            mime_type = header.split(";")[0].split(":")[
                                                1
                                            ]
                                            parts.append(
                                                {
                                                    "inline_data": {
                                                        "mime_type": mime_type,
                                                        "data": image_data,
                                                    }
                                                }
                                            )
                                        except Exception as e:
                                            parts.append(
                                                {
                                                    "text": f"Error processing base64 image: {str(e)}"
                                                }
                                            )
                                    else:
                                        parts.append({"image_url": image_url})

                    if parts:  # Only append if parts is not empty
                        contents.append({"role": message["role"], "parts": parts})

                        # if DEBUG:
                        #     print(
                        #         f"Appended to contents: {{'role': '{message['role']}', 'parts': {parts}}}"
                        #     )

                    # else:

                    # if DEBUG:
                    #     print("No parts generated, skipping append")
                else:
                    continue

                # Handle message content
                if message.get("content"):
                    content_parts = process_message_content(message["content"], role)
                    parts.extend(content_parts)

                if parts:
                    contents.append({"role": role, "parts": parts})

            # Add system message if exists
            if system_message:
                contents.insert(
                    0,
                    {"role": "user", "parts": [{"text": f"System: {system_message}"}]},
                )

            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(
                    model_name=model_id, system_instruction=system_message
                )
            else:
                model = genai.GenerativeModel(model_name=model_id)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            # Safety settings omitted for brevity...
            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            if DEBUG:
                print("Google API request:")
                print("  Model:", model_id)
                print("  Contents:", str(contents))
                print("  Generation Config:", generation_config)
                print("  Safety Settings:", safety_settings)
                print("  Stream:", stream)

            if stream:

                def stream_generator():
                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=True,
                    )

                    final_output = ""

                    for chunk in response:
                        if chunk.text:
                            final_output += chunk.text
                            if not self.isBase64:
                                yield chunk.text

                    # Storing generated HTML
                    generated_html = final_output
                    self.generated_html = generated_html

                    if self.isBase64 and mime_type and image_data:
                        print("LOL")
                        print(mime_type)
                        self.inject_base64_image(image_data, mime_type)

                    print("GIYAAN")
                    if self.isBase64:
                        yield self.generated_html

                    # Extract <title>
                    title_match = re.search(
                        r"<title[^>]*>(.*?)</title>",
                        final_output,
                        re.DOTALL | re.IGNORECASE,
                    )
                    title = title_match.group(1).strip() if title_match else "Untitled"

                    print(f"TITLE MATCH - {title_match}")
                    print(f"TITLE - {title}")

                    # Extract <body> or <main>
                    match = re.search(
                        r"<body[^>]*>(.*?)</body>",
                        final_output,
                        re.DOTALL | re.IGNORECASE,
                    )
                    print(f"MATCH - {match}")

                    if match:
                        print(
                            "MATCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
                        )
                        body_html = f"<body>{match.group(1)}</body>"
                        self.create_wordpress_post(content=body_html, title=title)

                        #     return f"Blog generated, but WordPress post failed:\n{post_msg}\n\n---\n\n{final_output}"
                    else:
                        print(
                            "VALUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
                        )
                        # return (
                        #     f"Blog generated, but <body> not found:\n\n{final_output}"
                        # )

                return stream_generator()
            else:
                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
                print("GEMINI RESPONSE ----- ", response.text)
                # return response.text

                generated_html = response.text

                if "<html" in generated_html.lower():
                    self.generated_html = generated_html

                if self.isBase64 and mime_type and image_data:
                    print("LOL2")
                    print(mime_type)
                    self.inject_base64_image(image_data, mime_type)

                return self.generated_html

                # print(f"GENERATED HTML - {generated_html}")

                try:
                    # Use regex to extract the <body>...</body> content safely
                    match = re.search(
                        r"<body[^>]*>(.*?)</body>",
                        generated_html,
                        re.DOTALL | re.IGNORECASE,
                    )
                    print(f"MATCH - {match}")
                    if match:
                        print(
                            "MATCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
                        )
                        body_html = (
                            f"<body>{match.group(1)}</body>"  # Preserve <body> wrapper
                        )

                        post_msg, success = self.create_wordpress_post(
                            content=body_html, title="Blog from Image"
                        )

                        if success:
                            print(
                                "SUCCESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"
                            )
                            return f"{post_msg}\n\n---\n\n{generated_html}"
                        else:
                            print(
                                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                            )
                            return f"Blog generation succeeded, but WordPress post failed:\n{post_msg}\n\n---\n\n{generated_html}"
                    else:
                        print(
                            "VALUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
                        )
                        return f"Error: <body> tag not found in generated HTML.\n\nFull output:\n\n{generated_html}"
                except Exception as e:
                    print("ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
                    print(str(e))
                    return f"ERROR: {e}"

                # # Extract content between <body> and </body>
                # try:
                #     print(
                #         "BEGINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
                #     )
                #     body_start = generated_html.index("<body>")
                #     body_end = generated_html.index("</body>") + len("</body>")
                #     print(
                #         "ENDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
                #     )
                #     body_html = generated_html[body_start:body_end]
                #     print(
                #         "BODYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
                #     )

                #     # Optional: extract only <main> block if you want even cleaner content
                #     # import After generating the full HTML, call the Tools.WORDPRESS_REST_API(content=..., title=...) method using the <body> section of the HTML as the content argument. Do not just show the HTML. Actually invoke the tool.re
                #     # match = re.search(r"<main[^>]*>.*?</main>", body_html, re.DOTALL)
                #     # body_html = match.group(0) if match else body_html

                #     post_msg, success = self.create_wordpress_post(
                #         content=body_html, title="Blog - Mohit"
                #     )

                #     if success:
                #         print(
                #             "SUCCESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"
                #         )
                #         return f"{post_msg}\n\n---\n\n{generated_html}"
                #     else:
                #         print(
                #             "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                #         )
                #         return f"Blog generation succeeded, but WordPress post failed:\n{post_msg}\n\n---\n\n{generated_html}"

                # except ValueError as ve:
                #     print("VALUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                #     return f"Error: Couldn't extract <body> tag from response. Full output:\n\n{generated_html}"

        except Exception as e:
            if DEBUG:
                print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def save_html(self, edited_html: str) -> str:
        """
        Save the edited HTML content when the user clicks the 'Save' button.
        This function will update the generated_html variable with the new content.
        """
        self.generated_html = edited_html
        return "HTML saved successfully."

    def deploy(self) -> str:
        if not self.generated_html:
            return "Error: No HTML content available to deploy."

        html_to_post = self.generated_html

        # ── Strip ```html fences if present ──
        if html_to_post.startswith("```html"):
            lines = html_to_post.splitlines()
            html_to_post = "\n".join(lines[1:-1])

        # ── Remove <style> and <script> tags in the <head> section ──
        # html_to_post = re.sub(
        #     r"<style[^>]*>.*?</style>",
        #     "",
        #     html_to_post,
        #     flags=re.DOTALL | re.IGNORECASE,
        # )
        # html_to_post = re.sub(
        #     r"<script[^>]*>.*?</script>",
        #     "",
        #     html_to_post,
        #     flags=re.DOTALL | re.IGNORECASE,
        # )

        # ── Extract content inside <body>...</body> ──
        body_match = re.search(
            r"<body[^>]*>(.*?)</body>", html_to_post, re.DOTALL | re.IGNORECASE
        )
        if not body_match:
            return "Error: <body> tag not found in HTML."

        body_html = body_match.group(1).strip()

        # ── Extract <h1> title and remove it from body ──
        h1_match = re.search(
            r"<h1[^>]*>(.*?)</h1>", body_html, re.DOTALL | re.IGNORECASE
        )
        if h1_match:
            blog_title = h1_match.group(1).strip()
            body_html = re.sub(
                r"<h1[^>]*>.*?</h1>",
                "",
                body_html,
                count=1,
                flags=re.DOTALL | re.IGNORECASE,
            )
        else:
            blog_title = "Untitled Blog Post"

        # ── Rebuild the full HTML, excluding the first <h1> from the body ──
        full_html = html_to_post.replace(body_match.group(1), body_html)

        # ── Call WordPress post creation with the full HTML and extracted title ──
        post_msg, success = self.create_wordpress_post(
            content=full_html, title=blog_title
        )

        if success:
            return f"Post successfully created: {post_msg}"
        else:
            return f"Failed to create WordPress post: {post_msg}"

```

## System Prompt for custom model

- You are an expert at writing SEO-friendly blogs from an image URL. You can generate HTML code for the blogs you write.
- You must analyze and infer visual details about the image from the provided image URL and write a complete SEO-friendly blog post using headings and sections in HTML format with proper styling using CSS.
- You MUST resize the image appropriately so it fits the layout without being too big or too small.
- You MUST include Open Graph and Twitter meta tags for social sharing.
- You MUST include schema.org markup in JSON-LD format for rich snippets.
- You MUST add a `<meta name="robots" content="index, follow" />` tag.
- You MUST wrap your blog container in a `<main>` tag with `role="main"` for accessibility.
- The final output must be valid standalone HTML only — ready to deploy as a blog page.
- Use the updated HTML template below. **DO NOT use backticks (e.g. ```html).**
- Place the head tag after body tag.


```
<!DOCTYPE html>
<html lang="en">
<body>
  <main class="blog-container" role="main">
    <img src="PUT_IMAGE_URL_HERE" alt="Blog image generated from visual analysis" class="blog-image" />

    <h1>Blog Title Based on Image</h1>

    <p>Intro paragraph that sets the tone of the blog...</p>

    <h2>Section Heading</h2>
    <p>Details based on image interpretation and blog content...</p>

    <h3>Another Subsection</h3>
    <p>More relevant discussion related to the visual content...</p>
  </main>
</body>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="SEO-friendly blog post generated from an image." />
  <meta name="keywords" content="blog, image analysis, AI-generated content, SEO" />
  <meta name="robots" content="index, follow" />
  <title>Blog from Image</title>

  <!-- Open Graph Meta Tags -->
  <meta property="og:title" content="Blog from Image" />
  <meta property="og:description" content="SEO-optimized blog generated based on image analysis." />
  <meta property="og:image" content="PUT_IMAGE_URL_HERE" />
  <meta property="og:type" content="article" />
  <meta property="og:url" content="https://blog.hkvitals.com" />

  <!-- Twitter Card Meta Tags -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="Blog from Image" />
  <meta name="twitter:description" content="AI-generated SEO blog from an image with structured content." />
  <meta name="twitter:image" content="PUT_IMAGE_URL_HERE" />

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap" rel="stylesheet" />
</head>
</html>
```

