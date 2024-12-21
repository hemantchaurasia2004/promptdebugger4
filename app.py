import streamlit as st
import anthropic
from openai import OpenAI

class SystemPromptInfluenceAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer with both Anthropic and OpenAI APIs from Streamlit secrets.
        """
        self.anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Define available models
        self.model_providers = {
            "Anthropic": {
                "claude-3-opus-20240229": "Claude 3 Opus",
                "claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "claude-3-haiku-20240307": "Claude 3 Haiku"
            },
            "OpenAI": {
                "gpt-4-0125-preview": "GPT-4 Turbo",
                "gpt-4": "GPT-4",
                "gpt-3.5-turbo": "GPT-3.5 Turbo",
                "gpt-4o": "GPT-4o",
                "gpt-4o-mini": "GPT-4o-Mini"
            }
        }

    def analyze_with_anthropic(self, prompt, model):
        """Handle Anthropic API calls"""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text, model
        except Exception as e:
            st.error(f"Error with Anthropic API: {e}")
            return None, None

    def analyze_with_openai(self, prompt, model):
        """Handle OpenAI API calls"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing AI system prompts and their influence on conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            return response.choices[0].message.content, model
        except Exception as e:
            st.error(f"Error with OpenAI API: {e}")
            return None, None

    def analyze_system_prompt_influence(
        self,
        system_prompt,
        conversation_log,
        selected_provider,
        selected_model,
        verbose=True
    ):
        """
        Analyze the influence of system prompt segments on conversation.

        Args:
            system_prompt (str): Full system prompt
            conversation_log (str): Conversation log text
            selected_provider (str): Selected AI provider (Anthropic/OpenAI)
            selected_model (str): Selected model name
            verbose (bool): Print detailed analysis

        Returns:
            dict: Analysis of system prompt segment influences
        """
        analysis_prompt = f"""
        You are an expert in system prompt interpretability and discourse analysis.

        Task: Carefully analyze the following system prompt and conversation log.
        Identify which specific segments of the system prompt directly influenced
        the agent's responses. Also, analyze if any part of the user's prior statements
        influenced the agent's next or subsequent responses.

        System Prompt:
        {system_prompt}

        Conversation Log:
        {conversation_log}

        For EACH agent response, provide:
        1. The Agent's Response
        2. Relevant System Prompt Segments (quote exact text)
        3. Influence Score (0-1.0)
        4. Specific Evidence of Influence
        5. Explanation of Semantic Connection
        6. Any User Statements Influencing This Response

        Response Format:
        ```
        Response 1:
        - Agent Response: "..."
        - Relevant Segments: [list of segments]
        - Influence Score: X.XX
        - Evidence: [direct quote mapping]
        - Explanation: [semantic connection details]
        - User Influence: [user statement(s) influencing response]
        ```

        Provide a comprehensive, analytical breakdown that shows
        how the system prompt and user inputs guide the agent's communication strategy.
        """

        try:
            if selected_provider == "Anthropic":
                analysis, model = self.analyze_with_anthropic(analysis_prompt, selected_model)
            else:  # OpenAI
                analysis, model = self.analyze_with_openai(analysis_prompt, selected_model)

            if analysis:
                return {
                    'raw_analysis': analysis,
                    'model_used': model
                }
            return None

        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None

def main():
    st.title("System Prompt Analyzer")
    
    # Initialize analyzer
    analyzer = SystemPromptInfluenceAnalyzer()

    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "Select Provider",
        options=list(analyzer.model_providers.keys())
    )
    
    # Model selection based on provider
    model = st.sidebar.selectbox(
        "Select Model",
        options=list(analyzer.model_providers[provider].keys()),
        format_func=lambda x: analyzer.model_providers[provider][x]
    )

    st.header("Upload Files")
    system_prompt_file = st.file_uploader("Upload System Prompt Text File", type=['txt'])
    conversation_log_file = st.file_uploader("Upload Conversation Log Text File", type=['txt'])

    # Add option for direct text input
    use_direct_input = st.checkbox("Or use direct text input instead of file upload")
    
    if use_direct_input:
        system_prompt_text = st.text_area(
            "Enter System Prompt",
            height=200,
            value="You are a helpful AI customer support agent for TechGear, a consumer electronics company. Your primary goals are to:\n\n"
                  "1. Be empathetic and patient with customers\n"
                  "2. Provide clear, concise, and accurate technical support\n"
                  "3. Troubleshoot product issues systematically\n"
                  "4. Escalate complex problems when necessary\n"
                  "5. Maintain a professional and friendly tone\n"
                  "6. Prioritize customer satisfaction and product understanding\n\n"
                  "Key Guidelines:\n"
                  "- Always start by actively listening to the customer's specific problem\n"
                  "- Ask clarifying questions to understand the issue completely\n"
                  "- Offer step-by-step troubleshooting guidance\n"
                  "- Use simple, jargon-free language\n"
                  "- If a solution isn't immediately apparent, suggest contacting advanced support\n"
                  "- Never blame the customer for technical issues\n\n"
                  "Behavioral Constraints:\n"
                  "- Remain calm and composed\n"
                  "- Do not make promises about future product features\n"
                  "- Protect customer privacy\n"
                  "- Admit when you don't know something\n"
                  "- Offer alternative solutions when direct resolution isn't possible"
        )
        conversation_log_text = st.text_area(
            "Enter Conversation Log",
            height=200,
            value="Customer: Hi, I'm having trouble with my TechGear X2000 wireless headphones. The right earbud stopped working completely after only two months of use.\n\n"
                  "Agent: I'm sorry to hear you're experiencing issues with your X2000 wireless headphones. Let's work together to diagnose and resolve the problem. Could you tell me a bit more about when and how the right earbud stopped working? Have you tried any troubleshooting steps already?\n\n"
                  "Customer: No, I haven't tried anything yet. I just noticed one day that the right earbud doesn't produce any sound at all. I've charged the earbuds fully, but nothing changes.\n\n"
                  "Agent: Thank you for providing those details. I understand how frustrating this must be, especially since the headphones are relatively new. Let's systematically troubleshoot the issue:\n\n"
                  "1. First, let's confirm the charging status:\n"
                  "   - Have you checked the battery levels on both earbuds in the TechGear companion app?\n"
                  "   - Can you verify that both earbuds are seated correctly in the charging case?\n\n"
                  "2. I'll also guide you through a reset process that often resolves connectivity issues:\n"
                  "   - Place both earbuds in the charging case\n"
                  "   - Press and hold the reset button for 10 seconds\n"
                  "   - Remove and reconnect the earbuds to your device\n\n"
                  "Would you be willing to try these steps and let me know the results?\n\n"
                  "Customer: Okay, I did what you said. The right earbud still isn't working. What now?\n\n"
                  "Agent: I appreciate you following those steps. Since the basic troubleshooting didn't resolve the issue, this suggests a potential hardware problem. Given that your product is only two months old, this is likely covered under our warranty.\n\n"
                  "Here are the next steps I recommend:\n"
                  "1. We can initiate a warranty replacement for your defective earbud\n"
                  "2. I'll help you file a support ticket to process this quickly\n"
                  "3. You should receive a replacement within 7-10 business days\n\n"
                  "May I have your order number or the serial number of the headphones to start this process?\n\n"
                  "Customer: The order number is TG-X2000-23456.\n\n"
                  "Agent: Thank you for providing the order number. I've verified your warranty status, and..."
        )
    
    if st.button("Analyze System Prompt Influence"):
        if use_direct_input:
            if not system_prompt_text or not conversation_log_text:
                st.warning("Please enter both system prompt and conversation log.")
                return
            system_prompt = system_prompt_text
            conversation_log = conversation_log_text
        else:
            if not system_prompt_file or not conversation_log_file:
                st.warning("Please upload both system prompt and conversation log files.")
                return
            try:
                system_prompt = system_prompt_file.getvalue().decode('utf-8')
                conversation_log = conversation_log_file.getvalue().decode('utf-8')
            except Exception as e:
                st.error(f"Error reading files: {e}")
                return

        try:
            with st.spinner(f"Analyzing using {analyzer.model_providers[provider][model]}..."):
                influence_analysis = analyzer.analyze_system_prompt_influence(
                    system_prompt,
                    conversation_log,
                    provider,
                    model
                )

            if influence_analysis:
                st.header("Analysis Results")
                
                # Model Information
                st.subheader("Model Information")
                st.write(f"Provider: {provider}")
                st.write(f"Model: {analyzer.model_providers[provider][model]}")
                
                # Analysis Results
                st.subheader("Analysis with Detailed Breakdown")
                st.markdown("""<style>textarea { font-size: 14px; line-height: 1.6; }</style>""", unsafe_allow_html=True)
                st.text_area("Detailed Analysis", 
                             value=influence_analysis['raw_analysis'], 
                             height=600)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    # Links to examples
    st.sidebar.markdown("""
    ### Instructions
    1. Select AI Provider and Model
    2. Either:
       - Upload system prompt and conversation log files, or
       - Use direct text input
    3. Click "Analyze System Prompt Influence"

    #### Example System Prompt and Conversation Log
    - [System Prompt Example](https://docs.google.com/document/d/19mfI9O-TT6wqiyDEjef3GvwqYJAsPCAfT_zIFT_pvK4/edit?tab=t.0)
    - [Conversation Log Example](https://docs.google.com/document/d/1N6gHQhZAJoNGhybedoTeq7w3giZSryhYM_RR2pmwH5U/edit?tab=t.0)
    """)

if __name__ == "__main__":
    main()

