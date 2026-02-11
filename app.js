// STORAGE
let users = JSON.parse(localStorage.getItem("users")) || {};
let cart = JSON.parse(localStorage.getItem("cart")) || [];

// PRODUCTS
const products = [
  { id: 1, name: "Wireless Headphones", price: 99, image: "🎧" },
  { id: 2, name: "Smart Watch", price: 199, image: "⌚" },
  { id: 3, name: "T-Shirt", price: 29, image: "👕" }
];

// HOME PAGE
const grid = document.getElementById("productsGrid");
if (grid) {
  grid.innerHTML = products.map(p => `
    <div class="product-card">
      <div class="product-image">${p.image}</div>
      <h3>${p.name}</h3>
      <p>$${p.price}</p>
      <button class="add-to-cart" onclick="addToCart(${p.id})">Add to Cart</button>
    </div>
  `).join("");
}

// ADD TO CART
function addToCart(id) {
  const product = products.find(p => p.id === id);
  cart.push(product);
  localStorage.setItem("cart", JSON.stringify(cart));
  alert("Added to cart");
}

// CART PAGE
const cartItems = document.getElementById("cartItems");
if (cartItems) {
  let total = 0;
  cartItems.innerHTML = cart.map(item => {
    total += item.price;
    return `<p>${item.name} - $${item.price}</p>`;
  }).join("");
  document.getElementById("cartTotal").textContent = total;
}

// REGISTER
const registerForm = document.getElementById("registerForm");
if (registerForm) {
  registerForm.onsubmit = e => {
    e.preventDefault();
    users[regEmail.value] = {
      name: regName.value,
      password: regPassword.value
    };
    localStorage.setItem("users", JSON.stringify(users));
    alert("Registered successfully!");
  };
}

// LOGIN
const loginForm = document.getElementById("loginForm");
if (loginForm) {
  loginForm.onsubmit = e => {
    e.preventDefault();
    const user = users[loginEmail.value];
    if (user && user.password === loginPassword.value) {
      alert("Login successful!");
      window.location.href = "index.html";
    } else {
      alert("Invalid credentials");
    }
  };
}
