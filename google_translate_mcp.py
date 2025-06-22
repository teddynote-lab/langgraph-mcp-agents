from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Optional
import httpx

# --- 환경 변수 로드 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Google Translate API Endpoint ---
TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

# --- Pydantic 모델 정의 ---
class TranslateRequest(BaseModel):
    """
    Google 번역을 위한 요청 모델입니다.
    """
    q: List[str] = Field(..., description="번역할 문자열 목록")
    target: str = Field(..., description="번역 대상 언어 (예: ko, en, ja)")
    source: Optional[str] = Field(None, description="원본 언어 (지정하지 않으면 자동 감지)")
    format: Optional[str] = Field("text", description="텍스트 형식 (text 또는 html)")
    model: Optional[str] = Field("nmt", description="번역 모델 (기본값: nmt)")

# --- 번역 함수 정의 ---
@mcp.tool()
async def translate_text(params: TranslateRequest):
    payload = {
        "q": params.q,
        "target": params.target,
        "format": params.format,
        "model": params.model,
        "key": GOOGLE_API_KEY,
    }
    if params.source:
        payload["source"] = params.source

    async with httpx.AsyncClient() as client:
        response = await client.post(TRANSLATE_URL, params=payload)
        response.raise_for_status()
        return response.json()

# --- FastMCP 서버 초기화 ---
mcp = FastMCP(
    "GoogleTranslator",
    instructions="너는 Google Translate API를 호출해서 번역 결과를 반환하는 도구야.",
    host="0.0.0.0",
    port=8000,
)

# --- MCP 명령 등록 ---
if __name__ == "__main__":
    # Run the MCP server with stdio transport for integration with MCP clients
    mcp.run(transport="stdio")
