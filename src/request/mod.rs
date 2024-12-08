use crate::{execution::logger::get_gtl_mut, program::Program};

use self::prompt::Prompt;

pub mod incoder;
pub mod openai;
pub mod prompt;

pub trait Handler {
    /// generate programs via a prefix string
    fn generate_by_str(&self, prefix: &str, suffix: &str) -> eyre::Result<Vec<Program>>;
    /// generate programs via a formatted prompt
    fn generate(&self, prompt: &Prompt) -> eyre::Result<Vec<Program>> {
        let start = std::time::Instant::now();
    
        // 새로운 방식으로 프롬프트를 구성
        let (user_msg_str, sys_msg_str) = prompt.to_completion_prompt();
    
        log::trace!("User message :{user_msg_str}");
        log::trace!("System message :{sys_msg_str}");
    
        // 사용자 메시지를 사용하여 프로그램 생성
        let mut programs = self.generate_by_str(&user_msg_str, &sys_msg_str)?;
    
        for program in programs.iter_mut() {
            program.combination = prompt.get_combination()?;
        }
    
        let generate_cost = start.elapsed().as_secs_f32();
        log::debug!("LLM Generate time: {}s", generate_cost);
        get_gtl_mut().inc_req(generate_cost);
    
        Ok(programs)
    }
    /// infill programs via the context of inserting location (i.e., prefix and suffix)
    fn infill_by_str(&self, prefix: &str, suffix: &str) -> eyre::Result<Vec<String>>;
    /// infill programs via a formatted prompt
    fn infill(&self, prompt: &Prompt) -> eyre::Result<Vec<String>> {
        let (prefix, suffix) = prompt.to_completion_prompt();
        self.infill_by_str(
            &prefix,
            &suffix,
        )
    }
    /// stop the service behind of the handler
    fn stop(&mut self) -> eyre::Result<()>;
}

/// format the url of server to send requests.
pub fn format_server_url() -> String {
    [
        "https://9a0c-34-125-12-196.ngrok-free.app",
    ]
    .join("")
}

/// concat the prefix, text and suffix.
pub fn program_infill(prefix: &str, text: &str, suffix: &str) -> String {
    let mut new_program = prefix.to_owned();
    new_program.push_str(text);
    new_program.push_str(suffix);
    new_program
}
