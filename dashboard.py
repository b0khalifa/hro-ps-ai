from datetime import datetime

import streamlit as st

from api_client import api_base_url, login_user_api
from approval_sections import show_admin_approval_panel
from audit_sections import show_audit_summary, show_audit_table, show_execution_trace
from dashboard_sections import (
    get_live_context,
    show_department_status,
    show_digital_twin,
    show_evaluation_panel,
    show_explainability_panel,
    show_forecast,
    show_operations_center,
    show_optimization,
    show_overview,
    show_simulation,
)
from message_center_sections import show_message_center
from notification_sections import show_notifications_panel
from staff_sections import (
    show_admin_appointments_overview,
    show_all_shifts,
    show_appointments,
    show_my_shifts,
    show_or_bookings,
)
from ui_components import inject_base_styles, sidebar_status_card

st.set_page_config(page_title="HRO Command Center", layout="wide")
inject_base_styles()

if "user" not in st.session_state:
    st.session_state.user = None


def login_view():
    st.title("🏥 HRO — AI Hospital System")
    st.caption("AI-powered hospital operations, forecasting, approvals, and staff coordination.")

    st.caption(f"API: {api_base_url()}")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user_api(username.strip(), password.strip()) if username.strip() and password.strip() else None
        if user:
            st.session_state.user = user
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")


def show_header(user):
    st.markdown(
        f"""
        <div class="hro-header">
            <div>👤 {user['name']} ({user['role']})</div>
            <div>🏥 HRO Command Center</div>
            <div>🕒 {datetime.now().strftime('%H:%M')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_navigation(role):
    st.sidebar.title("🧭 Navigation")

    if role == "admin":
        pages = [
            "Command Center",
            "Forecast",
            "Optimization",
            "Operations Center",
            "Shifts",
            "Appointments",
            "OR Bookings",
            "Notifications",
            "Messages",
            "Approvals",
            "Evaluation",
            "Explainability",
            "Audit",
        ]
    elif role == "doctor":
        pages = [
            "Overview",
            "Forecast",
            "My Shifts",
            "Appointments",
            "OR Bookings",
            "Notifications",
        ]
    else:
        pages = [
            "Overview",
            "My Shifts",
            "Appointments",
            "Department",
            "Notifications",
        ]

    return st.sidebar.radio("Go to", pages)


@st.cache_data(ttl=20, show_spinner=False)
def _cached_live_context():
    return get_live_context()


def show_sidebar_context(user):
    ctx = _cached_live_context()

    sidebar_status_card(
        "User Session",
        [
            f"<b>Name:</b> {user.get('name', '-')}",
            f"<b>Role:</b> {user.get('role', '-')}",
            f"<b>Department:</b> {user.get('department', '-')}",
        ],
    )

    if ctx.get("ready"):
        result = ctx["prediction_result"]
        sidebar_status_card(
            "Live Summary",
            [
                f"Current Patients: <b>{ctx['current_patients']}</b>",
                f"Next Hour Forecast: <b>{int(ctx['prediction'])}</b>",
                f"Peak Load: <b>{int(ctx['peak'])}</b>",
                f"Emergency: <b>{result.get('emergency_level', 'LOW')}</b>",
            ],
        )
    else:
        sidebar_status_card("System Status", [ctx.get("reason", "Context unavailable")])

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()


def main_app():
    user = st.session_state.user
    role = str(user["role"]).lower()

    show_header(user)
    show_sidebar_context(user)
    page = sidebar_navigation(role)

    if role == "admin":
        if page == "Command Center":
            show_overview()

        elif page == "Forecast":
            show_forecast()

        elif page == "Optimization":
            show_optimization()

        elif page == "Operations Center":
            st.markdown("## ⚙️ Operations Center")
            tab_ops, tab_sim, tab_twin, tab_dept = st.tabs(
                ["Operations", "Simulation", "Digital Twin", "Department Status"]
            )
            with tab_ops:
                show_operations_center()
            with tab_sim:
                show_simulation()
            with tab_twin:
                show_digital_twin()
            with tab_dept:
                show_department_status()

        elif page == "Shifts":
            show_all_shifts()

        elif page == "Appointments":
            show_admin_appointments_overview()

        elif page == "OR Bookings":
            show_or_bookings("admin")

        elif page == "Notifications":
            show_notifications_panel(user)

        elif page == "Messages":
            show_message_center(user)

        elif page == "Approvals":
            ctx = _cached_live_context()
            if not ctx["ready"]:
                st.error(ctx["reason"])
            else:
                result = ctx["prediction_result"]
                show_admin_approval_panel(
                    peak=int(ctx["peak"]),
                    beds_needed=int(result["recommended_resources"]["beds_needed"]),
                    doctors_needed=int(result["recommended_resources"]["doctors_needed"]),
                    emergency_level=result.get("emergency_level", "LOW"),
                    approver_name=user.get("name", "Admin"),
                )

        elif page == "Evaluation":
            show_evaluation_panel()

        elif page == "Explainability":
            show_explainability_panel()

        elif page == "Audit":
            show_audit_summary()
            st.markdown("---")
            show_audit_table()
            st.markdown("---")
            show_execution_trace()

    elif role == "doctor":
        if page == "Overview":
            show_overview()

        elif page == "Forecast":
            show_forecast()

        elif page == "My Shifts":
            show_my_shifts(user["username"], "doctor")

        elif page == "Appointments":
            show_appointments("doctor", doctor_name=user.get("name"))

        elif page == "OR Bookings":
            show_or_bookings("doctor", doctor_name=user.get("name"))

        elif page == "Notifications":
            show_notifications_panel(user)

    else:
        if page == "Overview":
            show_overview()

        elif page == "My Shifts":
            show_my_shifts(user["username"], "nurse")

        elif page == "Appointments":
            show_appointments("nurse", department=user.get("department"))

        elif page == "Department":
            show_department_status()

        elif page == "Notifications":
            show_notifications_panel(user)


if st.session_state.user is None:
    login_view()
else:
    main_app()