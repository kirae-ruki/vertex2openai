import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_vertex_models, get_vertex_express_models, refresh_models_config_cache
from credentials_manager import CredentialManager

router = APIRouter()

@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    await refresh_models_config_cache()
    
    PAY_PREFIX = "[PAY]"
    EXPRESS_PREFIX = "[EXPRESS] "
    OPENAI_DIRECT_SUFFIX = "-openai"
    OPENAI_SEARCH_SUFFIX = "-openaisearch"
    
    credential_manager_instance: CredentialManager = fastapi_request.app.state.credential_manager
    express_key_manager_instance = fastapi_request.app.state.express_key_manager

    has_sa_creds = credential_manager_instance.get_total_credentials() > 0
    has_express_key = express_key_manager_instance.get_total_keys() > 0

    raw_vertex_models = await get_vertex_models()
    raw_express_models = await get_vertex_express_models()
    
    final_model_list: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    current_time = int(time.time())

    def add_model_and_variants(base_id: str, prefix: str):
        """Adds a model and its variants to the list if not already present."""
        
        # Define all possible suffixes for a given model
        suffixes = [""] # For the base model itself
        
        is_gemini = "gemini" in base_id.lower()
        
        if is_gemini:
            # 只有 Gemini 模型才支持文本加密和 Auto 重试
            suffixes.extend(["-encrypt", "-encrypt-full", "-auto"])
            
            # OpenAI Direct 模式
            suffixes.append(OPENAI_DIRECT_SUFFIX)
            
            # 搜索特性（2.0以下模型基本不支持）
            if not base_id.startswith("gemini-2.0"):
                suffixes.extend(["-search", OPENAI_SEARCH_SUFFIX])
                
            # 思考特性（排除非 Thinking 的纯视觉或其他模型）
            if ("gemini-2.5-flash" in base_id or "gemini-2.5-pro" in base_id or "gemini-3-pro" in base_id or "gemini-3.1" in base_id) and "image" not in base_id:
                suffixes.extend(["-nothinking", "-max"])
                
            # 注: 原代码中的 -2k 和 -4k 因为在 chat_api.py 中没有做后缀剥离处理，
            # 会导致请求时直连报错“模型不存在”，所以本小姐替你把它们都砍掉了，保持列表干净！

        # Imagen 模型 (生图) 自动跳过上面的 if 判断，不添加任何乱七八糟的文本后缀，防止报错

        for suffix in suffixes:
            model_id_with_suffix = f"{base_id}{suffix}"
            
            # Experimental models have no prefix
            final_id = f"{prefix}{model_id_with_suffix}" if "-exp-" not in base_id else model_id_with_suffix

            if final_id not in processed_ids:
                final_model_list.append({
                    "id": final_id,
                    "object": "model",
                    "created": current_time,
                    "owned_by": "google",
                    "permission": [],
                    "root": base_id,
                    "parent": None
                })
                processed_ids.add(final_id)

    # Process Express Key models first
    if has_express_key:
        for model_id in raw_express_models:
            add_model_and_variants(model_id, EXPRESS_PREFIX)

    # Process Service Account (PAY) models, they have lower priority
    if has_sa_creds:
        for model_id in raw_vertex_models:
            add_model_and_variants(model_id, PAY_PREFIX)

    return {"object": "list", "data": sorted(final_model_list, key=lambda x: x['id'])}
