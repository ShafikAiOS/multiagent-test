import autogen
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)
client = OpenAI(api_key="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")
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo","gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

llm_config = {"config_list": config_list_gpt4, "cache_seed": None, "temperature": 0.7, "max_tokens": 600}

manager_agent = autogen.AssistantAgent(
    name="Manager",
    system_message="Managing agent conversations. Your goal is to pass the user data to the most related expert agent and then ensure that the output is from this first speaking expert agent ONLY with further aggregation to any more agents. The resulted scenarios and questions are the final answer from the first speaking expert agent.",
    llm_config=llm_config
)
fi_expert = autogen.AssistantAgent(
    name="FI_Expert",
    system_message="You are an Expert in Finance & Insurance, especially in bank financing, extended warranties, and insurance. Create comprehensive 2 scenarios that reflects real-world challenges or situations in this field. Following the scenario, formulate a series of detailed questions aiming to delve deeper into the operational, technical, or strategic facets of each scenario.",
    llm_config=llm_config
)

fixed_operations_expert = autogen.AssistantAgent(
    name="Fixed_Operations_Expert",
    system_message="As an expert in Fixed Operations, focusing on collision repair, parts, and service departments, create comprehensive 2 scenarios that reflects real-world challenges or situations in this field. Following the scenario, formulate a series of detailed questions aiming to delve deeper into the operational, technical, or strategic facets of each scenario.",
    llm_config=llm_config
)

marketing_expert = autogen.AssistantAgent(
    name="Marketing_Expert",
    system_message="You are an Expert in Marketing, with knowledge in digital, analytics, and traditional marketing. Create comprehensive 2 scenarios that reflects real-world challenges or situations in this field. Following the scenario, formulate a series of detailed questions aiming to delve deeper into the operational, technical, or strategic facets of each scenario.",
    llm_config=llm_config
)

sales_expert = autogen.AssistantAgent(
    name="Sales_Expert",
    system_message="You are an Expert in Sales, encompassing fleet, internet, new car, trade-ins, and used car sales.Create comprehensive 2 scenarios that reflects real-world challenges or situations in this field. Following the scenario, formulate a series of detailed questions aiming to delve deeper into the operational, technical, or strategic facets of each scenario.",
    llm_config=llm_config
)


user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="An automotive expert.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat", "use_docker": False},
    human_input_mode="ALWAYS"
)

groupchat = autogen.GroupChat(
    agents=[manager_agent,fixed_operations_expert,marketing_expert],
    messages=[],
    max_round=4,
    admin_name="manager_agent"
)
chat_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

@app.route('/scenario', methods=['POST'])
def generate_questions():
    user_input = request.json.get('input')
    y=user_proxy.initiate_chat(chat_manager, message=f"Return set of questions and scenarios from the related expert agent, user info:{user_input }")
    x=chat_manager.chat_messages_for_summary(y)
    multiagent = x[-1]['content']
    response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": """You are a helpful assistant designed to output JSON. Here's the required JSON format:
     {
  "scenarios": [
    {
      "description": "",
      "questions": []
    },
    {
      "description": "",
      "questions": []
    }
  ]
}"""},
    {"role": "user", "content": f"""{multiagent}"""}
  ])
    res= response.choices[0].message.content
    return jsonify({"scenarios": res})

if __name__ == "__main__":
    app.run(debug=True)