### ✅ 수정이 필요한 핵심: async context manager 사용 금지
### 🔁 MultiServerMCPClient를 사용할 때 async with 제거하고 get_tools()로 대체

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters import load_mcp_tools

async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.

    매개변수:
        mcp_config: MCP 도구 설정 정보(JSON). None인 경우 기본 설정 사용

    반환값:
        bool: 초기화 성공 여부
    """
    with st.spinner("🔄 MCP 서버에 연결 중..."):
        await cleanup_mcp_client()

        if mcp_config is None:
            mcp_config = load_config_from_json()

        try:
            # ❌ async with MultiServerMCPClient() 사용 금지
            client = MultiServerMCPClient(mcp_config)
            tools = await client.get_tools()  # 탐색이 가능한 MCP tool의 목록을 받아오는 옵션
        except Exception as e:
            st.error(f"MCP 연결 오류: {str(e)}")
            return False

        st.session_state.tool_count = len(tools)
        st.session_state.mcp_client = client

        # 선택된 목록에 따른 model 처리
        selected_model = st.session_state.selected_model

        if selected_model in [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]:
            model = ChatAnthropic(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        else:  # OpenAI
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )

        agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True
