function login() {
  const roll = document.getElementById("rollno").value.trim();
  const pass = document.getElementById("password").value.trim();
  const msg = document.getElementById("msg");

  // Your demo credentials
  const validRoll = "24CU0320034";
  const validPass = "1234";

  if (roll === "" || pass === "") {
    msg.style.color = "red";
    msg.textContent = "Enter Roll Number and Password";
    return;
  }

  if (roll === validRoll && pass === validPass) {
    msg.style.color = "green";
    msg.textContent = "Login successful...";

    setTimeout(() => {
      // Opens your existing bus tracking HTML file
      window.location.href = "bus-tracker-multilang (1).html";
    }, 800);

  } else {
    msg.style.color = "red";
    msg.textContent = "Invalid login details";
  }
}
