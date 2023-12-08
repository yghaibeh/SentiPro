/**
 * Sentiment Analysis Chat Application Script
 * Handles chat interactions, API requests, and local storage operations.
 */

// DOM elements
const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");

// Global variable
let userText = null;

// Function to get the value of a cookie by name
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


// Function to load data from local storage on page load
const loadDataFromLocalstorage = () => {
    // Load saved chats and theme from local storage and apply/add on the page
    const themeColor = localStorage.getItem("themeColor");

    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";

    const defaultText = `<div class="default-text">
                            <h1>SentiPro - Sentiment Analyser</h1>
                            <p>Express yourself, share your thoughts, and let AI understand your emotions!<br> Your chat history will be displayed here.<br> Muhammad Yaman Ghaibeh</p>
                        </div>`

    const savedChats = localStorage.getItem("all-chats");
    chatContainer.innerHTML = savedChats || defaultText;
    chatContainer.scrollTo(0, chatContainer.scrollHeight); // Scroll to bottom of the chat container
}

// Function to create a new chat element
const createChatElement = (content, className) => {
    // Create new div and apply chat, specified class and set html content of div
    const chatDiv = document.createElement("div");
    chatDiv.classList.add("chat", className);
    chatDiv.innerHTML = content;
    return chatDiv; // Return the created chat div
}

// Function to get chat response from the server
const getChatResponse = async (incomingChatDiv) => {
    const API_URL = "";
    const pElement = document.createElement("p");
    const progressBarsContainer = document.createElement("div");
    // Define the properties and data_handlers for the API request
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({
            prompt: userText,
        })
    }
    // Send POST request to API, get response and set the response as paragraph element text
    try {
        const response = await (await fetch(API_URL, requestOptions)).json();

        // Create progress bars based on sentiment values
        const sentimentValues = {
            positive: response.positive,
            neutral: response.neutral,
            negative: response.negative
        };

        const progressBox = incomingChatDiv.querySelector(".progress-box");

        for (const [sentiment, value] of Object.entries(sentimentValues)) {

            console.log(sentiment, value)

            const progressBarContainer = document.createElement("div");
            progressBarContainer.classList.add("w3-light-grey", "w3-xxlarge", "w3-round-xlarge");

            const progressBarInner = document.createElement("div");


            let size = "w3-large"

            if (sentiment === "positive") {
                progressBarInner.classList.add("w3-container", size, "w3-padding", "w3-round-xlarge","w3-green", "w3-center");
            } else if (sentiment === "neutral") {
                progressBarInner.classList.add("w3-container", size, "w3-padding", "w3-round-xlarge", "w3-blue", "w3-center");
            } else if (sentiment === "negative") {
                progressBarInner.classList.add("w3-container", size, "w3-padding", "w3-round-xlarge", "w3-red", "w3-center");
            }

            progressBarInner.style.height = "40px";
            progressBarInner.style.width = `${value}%`;
            progressBarInner.textContent = `${value}%`;

            progressBox.appendChild(progressBarInner);
            // Add a line break after each progress bar
            const lineBreak = document.createElement("br");
            progressBox.appendChild(lineBreak);
        }


    } catch (error) { // Add error class to the paragraph element and set error text
        console.log(error);
        pElement.classList.add("error");
        pElement.textContent = "Oops! Something went wrong while retrieving the response. Please try again.";
    }
    // Remove the typing animation, append the paragraph element and save the chats to local storage
    incomingChatDiv.querySelector(".typing-animation").remove();
    incomingChatDiv.querySelector(".chat-details").appendChild(pElement);
    localStorage.setItem("all-chats", chatContainer.innerHTML);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
}


// Function to display typing animation and initiate chat response
const showTypingAnimation = () => {
    // Display the typing animation and call the getChatResponse function
    const html = ` <div class="chat-content">
                        <div class="chat-details">
                        <img src="https://cdn-icons-png.flaticon.com/512/1985/1985500.png" alt="chatbot-img">
                        <div class="typing-animation">
                            <div class="typing-dot" style="--delay: 0.2s"></div>
                            <div class="typing-dot" style="--delay: 0.3s"></div>
                            <div class="typing-dot" style="--delay: 0.4s"></div>
                        </div>
                    </div>
                    <div class="progress-box" style="width: 100%; height: 150px;"></div>
                </div>`;

    // Create an incoming chat div with typing animation and append it to chat container
    const incomingChatDiv = createChatElement(html, "incoming");
    chatContainer.appendChild(incomingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);

    // Only call getChatResponse if it hasn't been called before in this function
    if (!showTypingAnimation.calledGetChatResponse) {
        showTypingAnimation.calledGetChatResponse = true;
        getChatResponse(incomingChatDiv);

        // Reset the flag after a delay to allow the function to be called again
        setTimeout(() => showTypingAnimation.calledGetChatResponse = false, 500);
    }
};


// Function to handle outgoing user chat
const handleOutgoingChat = () => {
    userText = chatInput.value.trim(); // Get chatInput value and remove extra spaces
    if(!userText) return; // If chatInput is empty return from here

    // Clear the input field and reset its height
    chatInput.value = "";
    chatInput.style.height = `${initialInputHeight}px`;

    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="https://cdn-icons-png.flaticon.com/512/666/666201.png" alt="user-img">
                        <p>${userText}</p>
                    </div>
                </div>`;

    // Create an outgoing chat div with user's message and append it to chat container
    const outgoingChatDiv = createChatElement(html, "outgoing");
    chatContainer.querySelector(".default-text")?.remove();
    chatContainer.appendChild(outgoingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
    setTimeout(showTypingAnimation, 500);
}


// Event listener for delete button
deleteButton.addEventListener("click", () => {
    // Remove the chats from local storage and call loadDataFromLocalstorage function
    if(confirm("Are you sure you want to delete all the chats?")) {
        localStorage.removeItem("all-chats");
        loadDataFromLocalstorage();
    }
});


// Event listener for theme button
themeButton.addEventListener("click", () => {
    // Toggle body's class for the theme mode and save the updated theme to the local storage
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
});


// Set initial input height for chat input
const initialInputHeight = chatInput.scrollHeight;


// Event listener for dynamically adjusting the height of the input field
chatInput.addEventListener("input", () => {
    // Adjust the height of the input field dynamically based on its content
    chatInput.style.height =  `${initialInputHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

// Event listener for handling Enter key press in the chat input
chatInput.addEventListener("keydown", (e) => {
    // If the Enter key is pressed without Shift and the window width is larger
    // than 800 pixels, handle the outgoing chat
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleOutgoingChat();
    }
});

// Load data from local storage on page load
loadDataFromLocalstorage();

// Event listener for the send button to handle outgoing chat
sendButton.addEventListener("click", handleOutgoingChat);