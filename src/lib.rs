use airtable_flows::create_record;
use async_openai::{
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionResponseFormat, ChatCompletionResponseFormatType,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use chrono::prelude::*;
use dotenv::dotenv;
use flowsnet_platform_sdk::logger;
use schedule_flows::{schedule_cron_job, schedule_handler};
use serde_json;
use std::collections::HashMap;
use std::env;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    let cron_time_with_date = get_cron_time_with_date();
    schedule_cron_job(cron_time_with_date, String::from("cron_job_evoked")).await;
}

#[schedule_handler]
async fn handler(body: Vec<u8>) {
    dotenv().ok();
    logger::init();
    let json_contents = include_str!("../rust_chapter.json");

    let data: Vec<String> = serde_json::from_str(json_contents).expect("failed to parse json");
    let mut count = 0;
    let mut chunk_count = 0;
    let chunks_len = data.len();
    for user_input in data {
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
        "As a highly skilled assistant, you are tasked with generating informative question and answer pairs from the provided text. Focus on crafting Q&A pairs that are relevant to the primary subject matter of the text. Your questions should be engaging and answers concise, avoiding details of specific examples that are not representative of the text's broader themes. Aim for a comprehensive understanding that captures the essence of the content without being sidetracked by less relevant details."
    .into());

    let user_input = format!("
    Here is the user input to work with:
    ---
    {}
    ---
    Your task is to dissect this text for its central themes and most significant details, crafting question and answer pairs that reflect the core message and primary content. Avoid questions about specific examples that do not contribute to the overall understanding of the subject. The questions should cover different types: factual, inferential, thematic, etc., and answers must be concise and pertinent to the text's main intent. Please generate as many relevant question and answers as possible, focusing on the significance and relevance of each to the text's main topic. Provide the results in the following JSON format:
    {{
        \"qa_pairs\": [
            {{
                \"question\": \"<Your question>\",
                \"answer\": \"<Your answer>\"
            }},
            // ... additional Q&A pairs based on text relevance
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
        // .model("gpt-3.5-turbo-1106")
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

fn get_cron_time_with_date() -> String {
    let now = Local::now();
    let now_minute = now.minute() + 2;
    format!(
        "{:02} {:02} {:02} {:02} *",
        now_minute,
        now.hour(),
        now.day(),
        now.month(),
    )
}
