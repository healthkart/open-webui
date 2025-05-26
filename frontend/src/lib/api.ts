export async function chatCompletions(
  model: string,
  messages: any[],
  chatId: string,
  id: string,
  sessionId: string,
  features: any,
  user: any
) {
  const formData = new FormData();
  formData.append("model", model);
  formData.append("messages", JSON.stringify(messages));
  formData.append("chat_id", chatId);
  formData.append("id", id);
  formData.append("session_id", sessionId);
  formData.append("features", JSON.stringify(features));
  formData.append("variables", JSON.stringify({ "{{CURRENT_USER}}": user?.email || "unknown" }));

  try {
    const response = await fetch("/api/chat/completions", {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        authorization: `Bearer ${localStorage.token}`
      },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Error response:", errorText);
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    return response;
  } catch (error) {
    console.error("Request failed:", error);
    throw error;
  }
} 