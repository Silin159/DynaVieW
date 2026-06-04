import os
from pathlib import Path
import base64
import json

from PIL import Image

import openai
from google import genai

def image_to_base64(image_path: Path) -> str:
    """Convert an image file to a Base64-encoded string."""
    image_path = Path(image_path)
    if not image_path.exists():
        return None  # Return None instead of raising error

    with open(image_path, "rb") as image_file:
        x = base64.b64encode(image_file.read()).decode("utf-8")
    if image_path.suffix == ".png":
        return f"data:image/png;base64,{x}"
    elif image_path.suffix.lower() in [".jpg", ".jpeg"]:
        return f"data:image/jpeg;base64,{x}"
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

class GPTAlignmentEvaluator:
    def __init__(self, model: str):
        self.model = model
        # Initialize other necessary components, e.g., tokenizer, API client, etc.
        if "gpt" in self.model:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please set the OPENAI_API_KEY environment variable.")
            self.client = openai.OpenAI(api_key=api_key)
        elif "gemini" in self.model:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Please specify the GENINI_API_KEY environment variable.")
            self.client = genai.Client(api_key=api_key)
    
    def evaluate_character_num(self, image_path, true_char_num: int) -> str:
        if "gpt" in self.model:
            return self._evaluate_character_num_gpt(image_path, true_char_num)
        elif "gemini" in self.model:
            return self._evaluate_character_num_gemini(image_path, true_char_num)

    def evaluate_character_attr(self, image_path, char_desc: dict) -> str:
        if "gpt" in self.model:
            return self._evaluate_character_attr_gpt(image_path, char_desc)
        elif "gemini" in self.model:
            return self._evaluate_character_attr_gemini(image_path, char_desc)
        
    def evaluate_entities(self, image_path, entities: list) -> str:
        if "gpt" in self.model:
            return self._evaluate_entities_gpt(image_path, entities)
        elif "gemini" in self.model:
            return self._evaluate_entities_gemini(image_path, entities)
        
    def evaluate_location(self, image_path, location: str) -> str:
        if "gpt" in self.model:
            return self._evaluate_location_gpt(image_path, location)
        elif "gemini" in self.model:
            return self._evaluate_location_gemini(image_path, location)
        
    def evaluate_time(self, image_path, time: str) -> str:
        if "gpt" in self.model:
            return self._evaluate_time_gpt(image_path, time)
        elif "gemini" in self.model:
            return self._evaluate_time_gemini(image_path, time)

    def _evaluate_character_num_gpt(self, image_path, true_char_num: int) -> str:
        prompt = 'How many characters are in this image? Only answer an Arabic number.'
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64(image_path),
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
        )

        res = response.choices[0].message.content

        try:
            predicted_char_num = int(res)
        except:
            transform_messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a converter. "
                        "Given English words for a number, respond ONLY with valid JSON "
                        'of the form {"number": <integer>}. '
                        "Use Arabic numerals. No extra keys. No text outside JSON."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": res
                        }
                    ]
                }
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=transform_messages,
                response_format={"type": "json_object"},  # forces valid JSON syntax
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            predicted_char_num = data.get("number")

        if predicted_char_num == true_char_num:
            return 1.0
        else:
            return 0.0
        
    def _evaluate_character_num_gemini(self, image_path, true_char_num: int) -> str:
        prompt = "How many characters are in this image? Only answer an Arabic number."
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image, prompt]
        )
        res = response.text
        if res is None:
            return None

        print(f"Gemini response for character number: {res}")
        if res.strip().isdigit():
            predicted_char_num = int(res.strip())
        else:
            transform_prompt = (
                "You are a converter. "
                "Given English words for a number, respond ONLY with valid JSON "
                'of the form {"number": <integer>}. '
                "Use Arabic numerals. No extra keys. No text outside JSON.\n"
                f"Here is the input: {res}"
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=[transform_prompt]
            )
            raw = response.text
            data = json.loads(raw)
            predicted_char_num = data.get("number")
        
        if predicted_char_num == true_char_num:
            return 1.0
        else:
            return 0.0

    def _evaluate_character_attr_gpt(self, image_path, char_desc: dict) -> str:
        prompt = "Character descriptions:\n"
        for name, desp in char_desc.items():
            prompt += f'{name}: {desp}\n'
        prompt += f'Do characters in this image fit into their descriptions? Only answer yes or no.'

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64(image_path),
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
        )

        res = response.choices[0].message.content

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        
    def _evaluate_character_attr_gemini(self, image_path, char_desc: dict) -> str:
        prompt = "Character descriptions:\n"
        for name, desp in char_desc.items():
            prompt += f'{name}: {desp}\n'
        prompt += f'Do characters in this image fit into their descriptions? Only answer yes or no.'

        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image, prompt]
        )
        res = response.text
        if res is None:
            return None

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        
    def _evaluate_entities_gpt(self, image_path, entities: list) -> str:
        match_count = 0
        total_count = 0
        for ent in entities:
            # skip character name entity
            ent_prompt = f'Does this image contain or imply \'{ent}\'? Only answer yes or no.'
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_base64(image_path),
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": ent_prompt
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100,
            )

            res = response.choices[0].message.content

            total_count += 1
            if res.lower().startswith("yes"):
                match_count += 1
        
        return float(match_count) / total_count
    
    def _evaluate_entities_gemini(self, image_path, entities: list) -> str:
        match_count = 0
        total_count = 0
        for ent in entities:
            ent_prompt = f'Does this image contain or imply \'{ent}\'? Only answer yes or no.'
            image = Image.open(image_path)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image, ent_prompt]
            )
            res = response.text
            if res is None:
                continue

            total_count += 1
            if res.lower().startswith("yes"):
                match_count += 1
        
        if total_count == 0:
            return None
        else:
            return float(match_count) / total_count
        
    def _evaluate_location_gpt(self, image_path, location: str) -> str:
        if any([location.startswith(letter) for letter in ["a", "e", "i", "o", "u"]]):
            art = "an"
        else:
            art = "a"
        prompt = f'Is this image taken at {art} {location}? Only answer yes or no.'

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64(image_path),
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
        )

        res = response.choices[0].message.content

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        
    def _evaluate_location_gemini(self, image_path, location: str) -> str:
        if any([location.startswith(letter) for letter in ["a", "e", "i", "o", "u"]]):
            art = "an"
        else:
            art = "a"
        prompt = f'Is this image taken at {art} {location}? Only answer yes or no.'

        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image, prompt]
        )
        res = response.text
        if res is None:
            return None

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        
    def _evaluate_time_gpt(self, image_path, time: str) -> str:

        assert time in ["early morning", "morning", "noon", "afternoon", "evening", "night"]
        if time in ["night", "noon"]:
            art = "at"
        else:
            art = "in the"

        prompt = f'Is this image taken {art} {time}? Only answer yes or no.'

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64(image_path),
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
        )

        res = response.choices[0].message.content

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        
    def _evaluate_time_gemini(self, image_path, time: str) -> str:

        assert time in ["early morning", "morning", "noon", "afternoon", "evening", "night"]
        if time in ["night", "noon"]:
            art = "at"
        else:
            art = "in the"

        prompt = f'Is this image taken {art} {time}? Only answer yes or no.'

        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image, prompt]
        )
        res = response.text
        if res is None:
            return None

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        

class GPTConsistencyEvaluator:
    def __init__(self, model: str):
        self.model = model
        # Initialize other necessary components, e.g., tokenizer, API client, etc.
        if "gpt" in self.model:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please set the OPENAI_API_KEY environment variable.")
            self.client = openai.OpenAI(api_key=api_key)
        elif "gemini" in self.model:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Please specify the GENINI_API_KEY environment variable.")
            self.client = genai.Client(api_key=api_key)

    # Define methods for consistency evaluation as needed
    def evaluate_character(self, image_paths: list, char_precense: list, char_description: dict):
        if "gpt" in self.model:
            return self._evaluate_character_gpt(image_paths, char_precense, char_description)
        elif "gemini" in self.model:
            return self._evaluate_character_gemini(image_paths, char_precense, char_description)
    
    def evaluate_location(self, image_paths: list, locations: list):
        if "gpt" in self.model:
            return self._evaluate_location_gpt(image_paths, locations)
        elif "gemini" in self.model:
            return self._evaluate_location_gemini(image_paths, locations)
        
    def evaluate_style(self, image_paths: list):
        # Implement style consistency evaluation logic
        if "gpt" in self.model:
            return self._evaluate_style_gpt(image_paths)
        elif "gemini" in self.model:
            return self._evaluate_style_gemini(image_paths)

    def _evaluate_character_gpt(self, image_paths: list, char_precense: list, char_description: dict):
        # Implement character consistency evaluation logic
        consistency_score = 0.0
        char_eval_count = 0
        for char, desp in char_description.items():
            appear_idxs = []
            for idx in range(len(image_paths)):
                if char in char_precense[idx]:
                    appear_idxs.append(idx)
            
            if len(appear_idxs) < 2:
                pass
            else:
                content = []
                for idx in appear_idxs:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                             "url": image_to_base64(Path(image_paths[idx])),
                             "detail": "high"
                            }
                        }
                    )
                
                consist_prompt = f'Do all these images contain the same charcater {char}: {desp}? Only answer yes or no.'
                content.append(
                    {
                        "type": "text",
                        "text": consist_prompt
                    }
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": content
                    }
                ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=100,
                )

                res = response.choices[0].message.content

                if res.lower().startswith("yes"):
                    consistency_score += 1.0
                else:
                    consistency_score += 0.0
                char_eval_count += 1
        
        if char_eval_count > 0:
            return consistency_score / char_eval_count

    def _evaluate_character_gemini(self, image_paths: list, char_precense: list, char_description: dict):
        # Implement character consistency evaluation logic
        consistency_score = 0.0
        char_eval_count = 0
        for char, desp in char_description.items():
            appear_idxs = []
            for idx in range(len(image_paths)):
                if char in char_precense[idx]:
                    appear_idxs.append(idx)
            
            if len(appear_idxs) < 2:
                pass
            else:
                content = []
                for idx in appear_idxs:
                    content.append(Image.open(image_paths[idx]))
                
                consist_prompt = f'Do all these images contain the same charcater {char}: {desp}? Only answer yes or no.'
                content.append(consist_prompt)

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=content
                )

                res = response.text
                if res is None:
                    continue

                if res.lower().startswith("yes"):
                    consistency_score += 1.0
                else:
                    consistency_score += 0.0
                char_eval_count += 1
        
        if char_eval_count > 0:
            return consistency_score / char_eval_count
        else:
            return None

    def _evaluate_location_gpt(self, image_paths: list, locations: list):
        consistency_score = 0.0
        loc_eval_count = 0
        loc_set = list(set(locations))
        for loc in loc_set:
            target_idxs = []
            for idx in range(len(image_paths)):
                if locations[idx] == loc:
                    target_idxs.append(idx)
        
            if len(target_idxs) < 2:
                pass
            else:
                content = []
                for idx in target_idxs:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                             "url": image_to_base64(Path(image_paths[idx])),
                             "detail": "high"
                            }
                        }
                    )
                
                consist_prompt = f'Are all these images taken at the same {loc}? Only answer yes or no.'
                content.append(
                    {
                        "type": "text",
                        "text": consist_prompt
                    }
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": content
                    }
                ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=100,
                )

                res = response.choices[0].message.content

                if res.lower().startswith("yes"):
                    consistency_score += 1.0
                else:
                    consistency_score += 0.0
                loc_eval_count += 1
        
        if loc_eval_count > 0:
            return consistency_score / loc_eval_count
        
    def _evaluate_location_gemini(self, image_paths: list, locations: list):
        consistency_score = 0.0
        loc_eval_count = 0
        loc_set = list(set(locations))
        for loc in loc_set:
            target_idxs = []
            for idx in range(len(image_paths)):
                if locations[idx] == loc:
                    target_idxs.append(idx)
        
            if len(target_idxs) < 2:
                pass
            else:
                content = []
                for idx in target_idxs:
                    content.append(Image.open(image_paths[idx]))
                
                consist_prompt = f'Are all these images taken at the same {loc}? Only answer yes or no.'
                content.append(consist_prompt)

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=content
                )

                res = response.text
                if res is None:
                    continue

                if res.lower().startswith("yes"):
                    consistency_score += 1.0
                else:
                    consistency_score += 0.0
                loc_eval_count += 1
        
        if loc_eval_count > 0:
            return consistency_score / loc_eval_count
        else:
            return None
        
    def _evaluate_style_gpt(self, image_paths: list):
        content = []
        for image in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": 
                    {
                     "url": image_to_base64(Path(image)),
                     "detail": "high"
                    }
                }
            )

        consist_prompt = f'Are all these images in the same style? Only answer yes or no.'
        content.append(
            {
                "type": "text",
                "text": consist_prompt
            }
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
        )
        res = response.choices[0].message.content
        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0
        
    def _evaluate_style_gemini(self, image_paths: list):
        content = []
        for image in image_paths:
            content.append(Image.open(image))
        
        consist_prompt = f'Are all these images in the same style? Only answer yes or no.'
        content.append(consist_prompt)

        response = self.client.models.generate_content(
            model=self.model,
            contents=content
        )

        res = response.text
        if res is None:
            return None

        if res.lower().startswith("yes"):
            return 1.0
        else:
            return 0.0