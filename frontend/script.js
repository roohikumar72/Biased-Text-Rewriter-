async function send() {
  const msg = document.getElementById("msg").value;
  if (!msg) return;

  const res = await fetch("http://127.0.0.1:5000/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: msg })
  });

  const data = await res.json();

  const messagesDiv = document.getElementById("messages");
  messagesDiv.innerHTML += `<p class="user"><b>You:</b> ${msg}</p>`;
  messagesDiv.innerHTML += `<p class="bot"><b>Bot:</b> Label - ${data.label}</p>`;
  if (data.rewritten)
    messagesDiv.innerHTML += `<p class="bot"><b>Bot:</b> Rewritten - ${data.rewritten}</p>`;

  document.getElementById("msg").value = "";
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
