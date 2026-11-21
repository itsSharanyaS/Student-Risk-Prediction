
import streamlit as st
import hashlib
import hmac


# =========================================================
# AUTHENTICATION CONFIGURATION
# =========================================================

USERS = {
    "admin": {
        "password": "Admin@123",
        "role": "Admin"
    },
    "faculty": {
        "password": "Faculty@123",
        "role": "Faculty"
    },
    "mentor": {
        "password": "Mentor@123",
        "role": "Mentor"
    }
}


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# =========================================================
# LOGIN FUNCTION
# =========================================================

def login_user(username, password):

    if username not in USERS:
        return False

    stored_password = USERS[username]["password"]

    return hmac.compare_digest(password, stored_password)


# =========================================================
# AUTHENTICATION UI
# =========================================================

def render_login():

    st.markdown("""
        <div style="text-align:center; padding:20px;">
            <h1>🎓 Student Risk Prediction</h1>
            <p>Academic Intelligence & Early Warning System</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.subheader("🔐 Secure Login")

        username = st.text_input("Username", key="login_username")

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):

            if login_user(username, password):

                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = USERS[username]["role"]

                st.success("Login successful!")
                st.rerun()

            else:
                st.error("Invalid username or password")


# =========================================================
# LOGOUT FUNCTION
# =========================================================

def logout_user():

    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None

    st.rerun()


# =========================================================
# AUTHENTICATION STATUS
# =========================================================

def is_authenticated():

    return st.session_state.get("authenticated", False)


# =========================================================
# CURRENT USER ROLE
# =========================================================

def get_user_role():

    return st.session_state.get("user_role", None)


# =========================================================
# CURRENT USERNAME
# =========================================================

def get_username():

    return st.session_state.get("username", None)

