import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types
from google import genai

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from message_processing import (
    create_gemini_prompt,
    ENCRYPTION_INSTRUCTIONS,
)
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call,
)
from openai_handler import OpenAIDirectHandler
from project_id_discovery import discover_project_id

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        credential_manager_instance = fastapi_request.app.state.credential_manager
        express_key_manager_instance = fastapi_request.app.state.express_key_manager
        
        OPENAI_DIRECT_SUFFIX = "-openai"
        OPENAI_SEARCH_SUFFIX = "-openaisearch"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        EXPRESS_PREFIX = "[EXPRESS] " 
        
        base_model_name = request.model 
        
        is_express_model_request = False
        if base_model_name.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            base_model_name = base_model_name[len(EXPRESS_PREFIX):]

        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]

        is_openai_direct_model = False
        is_openai_search_model = False
        
        if base_model_name.endswith(OPENAI_SEARCH_SUFFIX):
            is_openai_search_model = True
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_SEARCH_SUFFIX)]
        elif base_model_name.endswith(OPENAI_DIRECT_SUFFIX):
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_DIRECT_SUFFIX)]
            
        if EXPERIMENTAL_MARKER in base_model_name:
            is_openai_direct_model = True

        is_grounded_search = base_model_name.endswith("-search")
        if is_grounded_search: base_model_name = base_model_name[:-len("-search")]

        # ==========================================
        # 本小姐的专属 Imagen 4 拦截器 (全知视角监控版)
        # ==========================================
        if base_model_name.startswith("imagen-4"):
            import time
            import httpx
            import re  # [ENI 注入] 引入正则用于抓取比例
            from api_helpers import execute_with_retry
            from credentials_manager import _refresh_auth
            
            print(f"\n[Imagen 拦截器] 🚀 触发专属生图链路 | 目标模型: {base_model_name}")
            
            # 1. 从对话记录中抽离最后一条 User 文本作为生图提示词
            prompt_text = "A beautiful landscape"
            for msg in reversed(request.messages):
                if msg.role == "user":
                    if isinstance(msg.content, str):
                        prompt_text = msg.content
                    elif isinstance(msg.content, list):
                        text_parts = [p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
                        if text_parts: prompt_text = " ".join(text_parts)
                    break
            
            # -----------------------------------------------------
            # [ENI 专属智能比例嗅探器] 修复中英文冒号，精准提取！
            # -----------------------------------------------------
            target_aspect_ratio = "4:3" # 保持你原本代码里的默认值
            
            # 兼容英文半角 ":" 和中文全角 "："
            ar_match = re.search(r'(?i)(?:--ar\s+)?(1[:：]1|16[:：]9|9[:：]16|3[:：]4|4[:：]3)', prompt_text)
            if ar_match:
                raw_ar = ar_match.group(1)
                target_aspect_ratio = raw_ar.replace("：", ":") # 统一转换成 Google 能认的英文冒号
                print(f"[Imagen 拦截器] 📏 捕捉到隐式比例要求！已将画面比例切换为: {target_aspect_ratio}")
                # 提取后把它从发给 AI 的提示词里干净地抹除
                prompt_text = re.sub(r'(?i)(?:--ar\s+)?(1[:：]1|16[:：]9|9[:：]16|3[:：]4|4[:：]3)', '', prompt_text).strip()
            else:
                extra_params = getattr(request, "model_extra", {}) or {}
                size_param = extra_params.get("size")
                if size_param:
                    if size_param == "1024x1024": target_aspect_ratio = "1:1"
                    elif size_param == "1024x768": target_aspect_ratio = "4:3"
                    elif size_param == "768x1024": target_aspect_ratio = "3:4"
                    elif size_param in ["1:1", "9:16", "16:9", "3:4", "4:3"]:
                        target_aspect_ratio = size_param
                    print(f"[Imagen 拦截器] 📐 捕捉到前端 Size 参数，已转换为: {target_aspect_ratio}")
            
            print(f"[Imagen 拦截器] 📝 最终处理完成的生图提示词: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")
            
            # 2. 自动鉴权与组装端点
            headers = {"Content-Type": "application/json"}
            target_url = ""
            if is_express_model_request:
                print(f"[Imagen 拦截器] 🔑 鉴权通道: 正在使用 Express Key")
                if express_key_manager_instance.get_total_keys() == 0:
                    print("[Imagen 拦截器] ❌ 错误: 无可用 Express Key")
                    return JSONResponse(status_code=401, content=create_openai_error_response(401, "无可用 Express Key", "auth_error"))
                _, express_key = express_key_manager_instance.get_express_api_key()
                target_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{await discover_project_id(express_key)}/locations/us-central1/publishers/google/models/{base_model_name}:predict?key={express_key}"
            else:
                print(f"[Imagen 拦截器] 🛡️ 鉴权通道: 正在使用 Service Account")
                rotated_credentials, rotated_project_id = credential_manager_instance.get_credentials()
                if not rotated_credentials:
                    print("[Imagen 拦截器] ❌ 错误: 无可用 SA 凭证")
                    return JSONResponse(status_code=401, content=create_openai_error_response(401, "无可用 SA 凭证", "auth_error"))
                token = _refresh_auth(rotated_credentials)
                headers["Authorization"] = f"Bearer {token}"
                target_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{rotated_project_id}/locations/us-central1/publishers/google/models/{base_model_name}:predict"

            # 3. 硬编码参数
            import random # 引入随机魔法
            
            payload = {
                "instances": [{"prompt": prompt_text}],
                "parameters": {
                    "sampleCount": 4,
                    "seed": random.randint(1, 2147483647),
                    "aspectRatio": target_aspect_ratio,  # [ENI 注入] 动态比例！
                    "enhancePrompt": False,  # [ENI 注入] 绝对禁止 Google 篡改你的提示词！
                    "negativePrompt": "blurry, low quality, worst quality, low resolution, jpeg artifacts, pixelated, grainy, noise, deformed, mutated, ugly, disfigured, bad anatomy, extra limbs, missing limbs, fused fingers, extra fingers, poorly drawn hands, bad hands, distorted face, asymmetric face, deformed face, text, watermark, signature, logo, username, cropped, overexposed, underexposed",
                    "personGeneration": "allow_all",
                    "safetySetting": "block_only_high",
                    "addWatermark": False,
                    "language":"auto",
                    "sampleImageSize": "2k",
                    "outputOptions": {
                        "mimeType": "image/jpeg",
                        "compressionQuality": 85 
                    }
                }
            }

            # --- [ENI 补回发送端透视镜] 让你在 Docker 里清清楚楚看到发给 Google 的东西！ ---
            print(f"\n[Imagen 发送透视镜] 🔍 即将发送给 Google 的完整 Payload：\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")

            # 4. 执行请求并拼装前端假流式响应
            async def _call_imagen():
                print(f"[Imagen 拦截器] ⏳ 正在向 Google 发起算力请求 (4张2K图)...")
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(target_url, headers=headers, json=payload)
                    if resp.status_code != 200:
                        print(f"[Imagen 拦截器] ❌ Google API 报错: HTTP {resp.status_code} - {resp.text}")
                    resp.raise_for_status()
                    return resp.json()

            try:
                resp_json = await execute_with_retry(_call_imagen)
                
                # --- 本小姐的灵魂透视镜：打印除了 Base64 之外的所有 API 响应细节 ---
                import copy
                transparent_resp = copy.deepcopy(resp_json)
                if "predictions" in transparent_resp:
                    for p in transparent_resp["predictions"]:
                        if "bytesBase64Encoded" in p:
                            p["bytesBase64Encoded"] = "<Base64 数据过于庞大，已被本小姐物理屏蔽 🛡️>"
                
                print(f"\n[Imagen 接收透视镜] 🔍 Google API 完整返回参数与元数据：\n{json.dumps(transparent_resp, indent=2, ensure_ascii=False)}\n")
                # -----------------------------------------------------------

                predictions = resp_json.get("predictions", [])
                
                valid_b64_images = []
                filtered_count = 0
                
                for pred in predictions:
                    b64 = pred.get("bytesBase64Encoded", "")
                    if b64 and isinstance(b64, str) and len(b64) > 100:
                        valid_b64_images.append(b64)
                    else:
                        filtered_count += 1
                        
                print(f"[Imagen 拦截器] ✅ 数据清洗完成！总请求对象: {len(predictions)} | 有效成图: {len(valid_b64_images)} | 被安全拦截: {filtered_count}")
                
                md_images = []
                for idx, b64 in enumerate(valid_b64_images):
                    md_images.append(f"![Imagen {idx+1}](data:image/jpeg;base64,{b64})")
                
                if md_images:
                    final_content = "\n\n---\n\n".join(md_images)
                    if filtered_count > 0:
                        final_content += f"\n\n*(注：有 {filtered_count} 张图触碰安全红线被没收了。)*"
                else:
                    final_content = "⚠️ **生成失败或被全额拦截**：API 未返回有效的图像数据。"
                    
                response_id = f"chatcmpl-imagen-{int(time.time())}"

                if request.stream:
                    async def _imagen_fake_stream():
                        if not md_images:
                            err_msg = "⚠️ **生成失败**：未获取到有效图像。"
                            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': err_msg}, 'finish_reason': None}]})}\n\n"
                        else:
                            # 【本小姐的终极修复】：以“整张图片”为单位发送，绝对不切碎 Base64 字符串！
                            for idx, md_img in enumerate(md_images):
                                if idx > 0:
                                    # 发送图片之间的分隔符
                                    sep_chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": "\n\n---\n\n"}, "finish_reason": None}]}
                                    yield f"data: {json.dumps(sep_chunk)}\n\n"
                                
                                # 将一整张图作为一个 Chunk 发送，保证前端拿到的永远是合法的、闭合的图片标签
                                img_chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": md_img}, "finish_reason": None}]}
                                yield f"data: {json.dumps(img_chunk)}\n\n"
                                
                                # 【灵魂休眠】：发完一张图，强行让协程休息 0.2 秒。
                                # 既给网络缓冲区放行，又给前端留出了渲染这张完整图片的喘息时间！
                                await asyncio.sleep(0.2)
                            
                            if filtered_count > 0:
                                warn_chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": f"\n\n*(注：有 {filtered_count} 张图触碰安全红线被没收了)*"}, "finish_reason": None}]}
                                yield f"data: {json.dumps(warn_chunk)}\n\n"

                        final_chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        print("[Imagen 拦截器] 🎉 整图滴流式下发完毕！前端绝对不会卡了！")
                        
                    from fastapi.responses import StreamingResponse
                    return StreamingResponse(_imagen_fake_stream(), media_type="text/event-stream")
                else:
                    print("[Imagen 拦截器] 🎉 JSON 响应已返回前端！")
                    return JSONResponse(content={
                        "id": response_id, "object": "chat.completion", "created": int(time.time()), "model": request.model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}]
                    })
            except Exception as e:
                print(f"[Imagen 拦截器] 💥 发生致命错误: {str(e)}")
                return JSONResponse(status_code=500, content=create_openai_error_response(500, str(e), "imagen_error"))

        # ==========================================
        # 核心：智能识别 image 并配置
        # ==========================================
        is_image_model = "image" in request.model.lower()
        if is_image_model:
            is_openai_direct_model = False
            
        gen_config_dict = create_generation_config(request)

        is_thinking_capable = "gemini-2.5" in base_model_name or "gemini-3" in base_model_name

        if is_thinking_capable:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = True

        if is_image_model:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = False

        client_to_use = None

        if is_express_model_request:
            if express_key_manager_instance.get_total_keys() == 0:
                error_msg = f"Model '{request.model}' requires an Express API key, but none are configured."
                return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))

            total_keys = express_key_manager_instance.get_total_keys()
            for attempt in range(total_keys):
                key_tuple = express_key_manager_instance.get_express_api_key()
                if key_tuple:
                    original_idx, key_val = key_tuple
                    try:
                        if "gemini-2.5-pro" in base_model_name or "gemini-2.5-flash" in base_model_name:
                            project_id = await discover_project_id(key_val)
                            base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
                            client_to_use = genai.Client(
                                vertexai=True,
                                api_key=key_val,
                                http_options=types.HttpOptions(base_url=base_url)
                            )
                            client_to_use._api_client._http_options.api_version = None
                        else:
                            client_to_use = genai.Client(vertexai=True, api_key=key_val)
                        break 
                    except Exception as e:
                        client_to_use = None 
                else:
                    client_to_use = None

            if client_to_use is None: 
                return JSONResponse(status_code=500, content=create_openai_error_response(500, "All configured Express API keys failed.", "server_error"))
        
        else: 
            rotated_credentials, rotated_project_id = credential_manager_instance.get_credentials()
            
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                except Exception as e:
                    return JSONResponse(status_code=500, content=create_openai_error_response(500, str(e), "server_error"))
            else: 
                return JSONResponse(status_code=401, content=create_openai_error_response(401, "No SA credentials available.", "authentication_error"))

        if not is_openai_direct_model and client_to_use is None:
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        if is_openai_direct_model:
            if is_express_model_request:
                openai_handler = OpenAIDirectHandler(express_key_manager=express_key_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_express=True, is_openai_search=is_openai_search_model)
            else:
                openai_handler = OpenAIDirectHandler(credential_manager=credential_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_openai_search=is_openai_search_model)
        else: 
            current_prompt_func = create_gemini_prompt

            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                    gen_config_dict["tools"].append(search_tool)
                else:
                    gen_config_dict["tools"] = [search_tool]
            
            if not isinstance(gen_config_dict.get("thinking_config"), dict):
                gen_config_dict["thinking_config"] = {}

            if is_image_model:
                gen_config_dict["thinking_config"]["include_thoughts"] = False
            else:
                gen_config_dict["thinking_config"]["include_thoughts"] = True

            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
