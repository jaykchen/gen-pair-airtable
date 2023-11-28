use airtable_flows::create_record;
use async_openai::{
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionResponseFormat, ChatCompletionResponseFormatType,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use dotenv::dotenv;
use flowsnet_platform_sdk::logger;
use std::collections::HashMap;
use std::env;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
  let _ =  inner().await;
}

async fn inner() {
    dotenv().ok();
    logger::init();
    let file_path = "test.txt";
    let contents = match tokio::fs::read_to_string(file_path).await {
        Ok(contents) => contents,
        Err(e) => {
            log::error!("Failed to read file: {:?}", e);
            return; // Exit the function if we can't read the file
        }
    };
    
    let chunks = split_text_into_chunks(&contents);
    let chunks_len = chunks.len();
    let mut chunk_count = 0;
    let mut count = 0;
    for user_input in chunks {
        chunk_count += 1;
        match gen_pair(&user_input).await {
            Ok(Some(qa_pairs)) => {
                for _ in qa_pairs {
                    count += 1;
                }
            }
            Ok(None) => {
                log::warn!("No Q&A pairs generated for the current chunk.");
            }
            Err(e) => {
                log::error!("Failed to generate Q&A pairs: {:?}", e);
            }
        }
        log::info!(
            "Processed {} Q&A pairs in {} of {} sections.",
            count,
            chunk_count,
            chunks_len
        );
    }
}

pub async fn gen_pair(
    user_input: &str,
) -> Result<Option<Vec<(String, String)>>, Box<dyn std::error::Error>> {
    let sys_prompt = env::var("SYS_PROMPT").unwrap_or(
    "As a highly skilled assistant, you are tasked with generating as many as possible informative question and answer pairs from the provided text. Craft Q&A pairs that are relevant, accurate, and varied in type (factual, inferential, thematic). Your questions should be engaging, and answers should be concise, both reflecting the text's intent. Aim for a comprehensive dataset that is rich in content and suitable for training language models, balancing the depth and breadth of information without redundancy."
.into());

    let user_input = format!("
Here is the user input to work with:

---
{}
---

Your task is to dissect this text for both granular details and broader themes, crafting as many Q&A pairs as possible. The questions should cover different types: factual, inferential, thematic, etc. Answers must be concise and reflective of the text's intent. Please generate as many question and answers as possible. Provide the results in the following JSON format:
{{
    \"qa_pairs\": [
        {{
            \"question\": \"<Your question>\",
            \"answer\": \"<Your answer>\"
        }},
        // ... additional Q&A pairs based on text length
    ]
}}",
        user_input
    );

    let messages = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(&sys_prompt)
            .build()
            .expect("Failed to build system message")
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_input)
            .build()?
            .into(),
    ];

    let client = Client::new();

    let response_format = ChatCompletionResponseFormat {
        r#type: ChatCompletionResponseFormatType::JsonObject,
    };

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(4000u16)
        .model("gpt-4-1106-preview")
        .messages(messages)
        .response_format(response_format)
        .build()?;

    let chat = match client.chat().create(request).await {
        Ok(chat) => chat,

        Err(e) => {
            log::error!("Failed to create chat: {:?}", e);
            return Ok(None);
        }
    };

    #[derive(serde::Deserialize)]
    struct QaPair {
        question: String,
        answer: String,
    }

    let mut qa_pairs_vec = Vec::new();
    if let Some(qa_pairs_json) = &chat.choices[0].message.content {
        let deserialized: HashMap<String, Vec<QaPair>> = match serde_json::from_str(&qa_pairs_json)
        {
            Ok(deserialized) => deserialized,
            Err(e) => {
                log::error!("Failed to deserialize qa_pairs_json: {:?}", e);
                return Ok(None);
            }
        };

        if let Some(qa_pairs) = deserialized.get("qa_pairs") {
            qa_pairs_vec = qa_pairs
                .iter()
                .map(|qa| (qa.question.clone(), qa.answer.clone()))
                .collect();
        }
    }
    for (question, answer) in &qa_pairs_vec {
        upload_airtable(question, answer).await;
    }

    Ok(Some(qa_pairs_vec))
}

pub fn split_text_into_chunks(raw_text: &str) -> Vec<String> {
    let mut res = Vec::new();
    let mut current_section = String::new();

    for line in raw_text.lines() {
        if !line.trim().is_empty() {
            current_section.push_str(line);
            current_section.push('\n');
        }

        if line.trim().is_empty() && !current_section.trim().is_empty() {
            res.push(current_section.clone());
            current_section.clear();
        }
    }
    res
}

pub async fn upload_airtable(question: &str, answer: &str) {
    let airtable_token_name = env::var("airtable_token_name").unwrap_or("github".to_string());
    let airtable_base_id = env::var("airtable_base_id").unwrap_or("appmhvMGsMRPmuUWJ".to_string());
    let airtable_table_name = env::var("airtable_table_name").unwrap_or("mention".to_string());

    let data = serde_json::json!({
        "Question": question,
        "Answer": answer,
    });
    let _ = create_record(
        &airtable_token_name,
        &airtable_base_id,
        &airtable_table_name,
        data.clone(),
    );
}
