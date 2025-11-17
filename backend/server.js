// server.js
const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
const PORT = 5001;

// --- NEW DEBUG MIDDLEWARE ---
// This will run for EVERY request that hits the server (GET, POST, OPTIONS, etc.)
app.use((req, res, next) => {
  console.log(`[Node DEBUG] Checkpoint 1: Received request: ${req.method} ${req.url}`);
  next(); // Continue to the next middleware (CORS)
});


// --- CORS Configuration ---
const corsOptions = {
  origin: ["http://localhost:5173", "https://nutriscense-frontend.netlify.app"],
  methods: "GET,HEAD,PUT,PATCH,POST,DELETE,OPTIONS",
  allowedHeaders: "Origin, X-Requested-With, Content-Type, Accept, Authorization",
  credentials: true,
};

app.use(cors(corsOptions));
app.use(express.json());

// --- Your API Route with a new debug log ---
app.post("/api/recipe", async (req, res) => {
  console.log("[Node DEBUG] Checkpoint 2: POST /api/recipe hit. Forwarding to Python...");
  try {
    const response = await axios.post("https://huggingface.co/spaces/text-ashish/ai_service/get_recipe", req.body);
    console.log("[Node DEBUG] Checkpoint 3: Received response from Python successfully.");
    res.json(response.data);
  } catch (err) {
    console.error("❌ Error forwarding request to Python service:", err.response ? err.response.data : err.message);
    res.status(500).json({ error: "The AI service failed to respond.", details: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Node.js proxy server running on http://localhost:${PORT}`);
});