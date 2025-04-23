from flask import Flask, render_template, request, jsonify
import requests
import time
import base64

app = Flask(__name__)
DEEPINFRA_API_KEY = "your_key"
TEXT_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
HF_API_KEY = "your_key"  
HF_IMAGE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

def generate_content(topic, tool_type):
    """Generate content using DeepInfra"""
    tool_prompts = {
        "article": f"Write a comprehensive article about: {topic}. Include introduction, body with facts, and conclusion. Also suggest an appropriate image description for this topic.",
        "blog": f"Write a conversational blog post about: {topic}. Use engaging tone with examples. Include an image description that would complement this blog post.",
        "rewrite": f"Rewrite this text to be unique while preserving meaning: {topic}",
        "paragraph": f"Write a detailed paragraph about: {topic}"
    }

    prompt = tool_prompts.get(tool_type, f"Write about {topic}")

    try:
        response = requests.post(
            TEXT_API_URL,
            headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
            json={
                "model": "meta-llama/Llama-2-70b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_image_hf(prompt):
    """Generate image using Hugging Face"""
    try:
        prompt = f"{prompt[:400]}, high resolution, 4K, digital art" if prompt else "digital art"
        
        response = requests.post(
            HF_IMAGE_API_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "url": None,
                "base64": f"data:image/png;base64,{base64.b64encode(response.content).decode('utf-8')}"
            }
        else:
            print(f"Hugging Face Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Hugging Face API Exception: {str(e)}")
        return None

def generate_image_replicate(prompt):
    """Generate image using Replicate (fallback)"""
    try:
        REPLICATE_API_KEY = "your_key"
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {REPLICATE_API_KEY}"},
            json={
                "version": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {"prompt": prompt[:500]}
            }
        )
        
        if response.status_code == 201:
            prediction_id = response.json()['id']
            for _ in range(10):  # Check for 20 seconds max
                prediction = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {REPLICATE_API_KEY}"}
                ).json()
                if prediction['status'] == 'succeeded':
                    return {"url": prediction['output'][0], "base64": None}
                time.sleep(2)
        return None
    except Exception as e:
        print(f"Replicate API Exception: {str(e)}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate/<tool_type>")
def generate_page(tool_type):
    return render_template("content-generator.html", tool_type=tool_type)

@app.route("/generate-all", methods=["POST"])
def generate_all():
    data = request.json
    topic = data.get("topic", "")
    tool_type = data.get("tool_type", "article")

    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400
    
    try:
        content = generate_content(topic, tool_type)
        if "error" in content.lower():
            return jsonify({"error": content}), 500
            
        image_prompt = topic
        if content and len(content.split('.')) > 1:
            image_prompt = content.split('.')[-2] + " realistic high quality"
        
        # Try Hugging Face first
        image_data = generate_image_hf(image_prompt)
        
        # If HF fails, try Replicate
        if not image_data:
            image_data = generate_image_replicate(image_prompt)
        
        # If both fail
        if not image_data:
            return jsonify({
                "content": content,
                "error": "Image generation failed after multiple attempts"
            })
        
        return jsonify({
            "content": content,
            "image_url": image_data.get("url"),
            "image_base64": image_data.get("base64")
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)