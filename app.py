import streamlit as st
import cv2
import math
import threading
from collections import defaultdict, Counter

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CrowdVision AI",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================
st.markdown("""
<style>

.stApp {
    background-color: #0b1020;
    color: white;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
}

.subtitle {
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 25px;
}

section[data-testid="stSidebar"] {
    background-color: #101629;
}

div[data-testid="stMetric"] {
    background-color: #151c31;
    border: 1px solid #29324d;
    border-radius: 14px;
    padding: 12px;
}

div[data-testid="stMetricValue"] {
    color: white;
    font-size: 30px;
}

div[data-testid="stMetricLabel"] {
    color: #b8c0d4;
}

.alert-box {
    background-color: #451a1a;
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 14px;
    color: #fecaca;
    font-weight: bold;
}

.info-box {
    background-color: #13233d;
    border: 1px solid #315b8c;
    border-radius: 10px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOLO POSE MODEL
# ============================================================
@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")


model = load_model()


# ============================================================
# SHARED LIVE STATE
# ============================================================
class LiveState:

    def __init__(self):

        self.lock = threading.Lock()

        self.data = {
            "People": 0,
            "Walking": 0,
            "Running": 0,
            "Standing": 0,
            "Sitting": 0,
            "Hand Raised": 0,
            "Fall": 0
        }

        self.persons = []


@st.cache_resource
def get_live_state():

    return LiveState()


live_state = get_live_state()


# ============================================================
# POSE POINT FUNCTION
# ============================================================
def get_point(kpts, index):

    try:

        x = float(kpts[index][0])
        y = float(kpts[index][1])
        confidence = float(kpts[index][2])

        if confidence < 0.25:
            return None

        return x, y

    except:

        return None


# ============================================================
# DISTANCE
# ============================================================
def distance(a, b):

    if a is None or b is None:
        return 0

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


# ============================================================
# COCO KEYPOINT INDEX
# ============================================================
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14


# ============================================================
# ACTION DETECTION
# ============================================================
def get_action(track_id, kpts, history):

    ls = get_point(kpts, LEFT_SHOULDER)
    rs = get_point(kpts, RIGHT_SHOULDER)

    lw = get_point(kpts, LEFT_WRIST)
    rw = get_point(kpts, RIGHT_WRIST)

    lh = get_point(kpts, LEFT_HIP)
    rh = get_point(kpts, RIGHT_HIP)

    lk = get_point(kpts, LEFT_KNEE)
    rk = get_point(kpts, RIGHT_KNEE)


    # --------------------------------------------------------
    # No shoulder information
    # --------------------------------------------------------
    if ls is None or rs is None:

        return "Standing"


    shoulder_y = (
        ls[1] + rs[1]
    ) / 2


    # ========================================================
    # HAND RAISED
    # ========================================================
    if lw is not None:

        if lw[1] < shoulder_y - 30:

            return "Hand Raised"


    if rw is not None:

        if rw[1] < shoulder_y - 30:

            return "Hand Raised"


    # ========================================================
    # FALL
    # ========================================================
    if lh is not None and rh is not None:

        shoulder_width = distance(
            ls,
            rs
        )

        body_height = abs(
            ((lh[1] + rh[1]) / 2)
            -
            ((ls[1] + rs[1]) / 2)
        )

        if body_height > 0:

            if shoulder_width > body_height * 1.5:

                return "Fall"


    # ========================================================
    # SITTING
    # ========================================================
    if (
        lh is not None
        and rh is not None
        and lk is not None
        and rk is not None
    ):

        hip_y = (
            lh[1] + rh[1]
        ) / 2

        knee_y = (
            lk[1] + rk[1]
        ) / 2

        difference = abs(
            knee_y - hip_y
        )

        if difference < 80:

            return "Sitting"


    # ========================================================
    # PERSON CENTER
    # ========================================================
    center_x = (
        ls[0] + rs[0]
    ) / 2

    center_y = (
        ls[1] + rs[1]
    ) / 2


    history[track_id].append(
        (center_x, center_y)
    )


    # Keep recent positions
    if len(history[track_id]) > 12:

        history[track_id] = (
            history[track_id][-12:]
        )


    positions = history[track_id]


    # ========================================================
    # MOVEMENT
    # ========================================================
    movement = 0


    if len(positions) >= 5:

        old = positions[0]
        new = positions[-1]

        movement = distance(
            old,
            new
        )


    # ========================================================
    # RUNNING
    # ========================================================
    if movement > 50:

        return "Running"


    # ========================================================
    # WALKING
    # ========================================================
    if movement > 10:

        return "Walking"


    # ========================================================
    # STANDING
    # ========================================================
    return "Standing"


# ============================================================
# PROCESS FRAME
# ============================================================
def process_frame(
    frame,
    confidence,
    history,
    show_ids
):

    result = model.track(
        frame,
        persist=True,
        conf=confidence,
        imgsz=640,
        verbose=False
    )[0]


    # YOLO annotations
    annotated = result.plot()


    people = []


    # ========================================================
    # NO PEOPLE
    # ========================================================
    if result.keypoints is None:

        return annotated, people


    keypoints = (
        result.keypoints.data
        .cpu()
        .numpy()
    )


    # ========================================================
    # TRACKING IDs
    # ========================================================
    if result.boxes.id is not None:

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

    else:

        ids = list(
            range(len(keypoints))
        )


    # ========================================================
    # PROCESS EACH PERSON
    # ========================================================
    for kpts, track_id in zip(
        keypoints,
        ids
    ):

        track_id = int(track_id)


        action = get_action(
            track_id,
            kpts,
            history
        )


        people.append({
            "id": track_id,
            "action": action
        })


        # ----------------------------------------------------
        # Find top-left position
        # ----------------------------------------------------
        valid_points = []


        for p in kpts:

            if len(p) >= 3:

                if p[2] > 0.25:

                    valid_points.append(
                        (
                            int(p[0]),
                            int(p[1])
                        )
                    )


        if not valid_points:

            continue


        x = min(
            p[0]
            for p in valid_points
        )

        y = min(
            p[1]
            for p in valid_points
        )


        # ====================================================
        # PERSON LABEL
        # ====================================================
        if show_ids:

            label = (
                f"ID {track_id} | {action}"
            )

        else:

            label = action


        # Background
        cv2.rectangle(
            annotated,
            (
                x - 5,
                max(0, y - 32)
            ),
            (
                x + 230,
                y + 5
            ),
            (20, 25, 40),
            -1
        )


        # Text
        cv2.putText(
            annotated,
            label,
            (
                x,
                max(22, y - 9)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


    return annotated, people


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">'
    '👁️ CrowdVision AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Human Pose Estimation & Action Recognition '
    'in Crowded Scenes'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.header("⚙️ Control Panel")


    mode = st.radio(
        "Select Mode",
        [
            "📷 Live Webcam",
            "📹 Upload Video"
        ]
    )


    st.divider()


    confidence = st.slider(
        "Detection Confidence",
        0.20,
        0.90,
        0.35,
        0.05
    )


    show_ids = st.checkbox(
        "Show Tracking IDs",
        True
    )


    st.divider()


    st.info(
        "YOLO Pose + Multi-Person Tracking "
        "is used for live analysis."
    )


# ============================================================
# DASHBOARD TITLE
# ============================================================
st.subheader("📊 Live Dashboard")


# ============================================================
# DASHBOARD
# ============================================================
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)


people_metric = c1.empty()
walking_metric = c2.empty()
running_metric = c3.empty()
standing_metric = c4.empty()
sitting_metric = c5.empty()
hand_metric = c6.empty()
fall_metric = c7.empty()


# ============================================================
# UPDATE DASHBOARD
# ============================================================
def update_dashboard(data):

    people_metric.metric(
        "👥 People",
        data["People"]
    )


    walking_metric.metric(
        "🚶 Walking",
        data["Walking"]
    )


    running_metric.metric(
        "🏃 Running",
        data["Running"]
    )


    standing_metric.metric(
        "🧍 Standing",
        data["Standing"]
    )


    sitting_metric.metric(
        "🪑 Sitting",
        data["Sitting"]
    )


    hand_metric.metric(
        "🙋 Hand Raised",
        data["Hand Raised"]
    )


    fall_metric.metric(
        "🚨 Falls",
        data["Fall"]
    )


# ============================================================
# INITIAL DASHBOARD
# ============================================================
update_dashboard({
    "People": 0,
    "Walking": 0,
    "Running": 0,
    "Standing": 0,
    "Sitting": 0,
    "Hand Raised": 0,
    "Fall": 0
})


# ============================================================
# LIVE WEBCAM
# ============================================================
if mode == "📷 Live Webcam":

    st.subheader(
        "📷 Live Webcam Monitoring"
    )


    st.markdown(
        '<div class="info-box">'
        'Click <b>START</b>, select your webcam, '
        'and allow camera permission in the browser.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # VIDEO PROCESSOR
    # ========================================================
    class CrowdProcessor(
        VideoProcessorBase
    ):

        def __init__(self):

            self.history = defaultdict(list)


        def recv(self, frame):

            img = frame.to_ndarray(
                format="bgr24"
            )


            # ------------------------------------------------
            # YOLO + POSE + TRACKING
            # ------------------------------------------------
            annotated, people = process_frame(
                img,
                confidence,
                self.history,
                show_ids
            )


            # ------------------------------------------------
            # CURRENT ACTION COUNTS
            # ------------------------------------------------
            counts = Counter(
                person["action"]
                for person in people
            )


            current_data = {

                "People":
                    len(people),

                "Walking":
                    counts["Walking"],

                "Running":
                    counts["Running"],

                "Standing":
                    counts["Standing"],

                "Sitting":
                    counts["Sitting"],

                "Hand Raised":
                    counts["Hand Raised"],

                "Fall":
                    counts["Fall"]
            }


            # ------------------------------------------------
            # SAVE LIVE DATA
            # ------------------------------------------------
            with live_state.lock:

                live_state.data = (
                    current_data.copy()
                )

                live_state.persons = (
                    people.copy()
                )


            # ------------------------------------------------
            # RETURN VIDEO
            # ------------------------------------------------
            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24"
            )


    # ========================================================
    # START WEBCAM
    # ========================================================
    ctx = webrtc_streamer(

        key="crowdvision-camera-final",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=CrowdProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    st.divider()


    # ========================================================
    # LIVE DASHBOARD REFRESH
    # ========================================================
    @st.fragment(run_every=0.5)
    def live_dashboard():

        with live_state.lock:

            data = (
                live_state.data.copy()
            )

            persons = (
                live_state.persons.copy()
            )


        # ----------------------------------------------------
        # UPDATE TOP CARDS
        # ----------------------------------------------------
        update_dashboard(data)


        # ----------------------------------------------------
        # PERSON STATUS
        # ----------------------------------------------------
        st.markdown(
            "### 👤 Person-wise Live Status"
        )


        if persons:

            number_of_columns = min(
                len(persons),
                4
            )


            person_cols = st.columns(
                number_of_columns
            )


            for i, person in enumerate(persons):

                person_cols[
                    i % number_of_columns
                ].info(
                    f"🆔 ID {person['id']}\n\n"
                    f"**{person['action']}**"
                )

        else:

            st.caption(
                "No person detected..."
            )


        # ----------------------------------------------------
        # FALL ALERT
        # ----------------------------------------------------
        if data["Fall"] > 0:

            st.markdown(
                '<div class="alert-box">'
                '🚨 FALL ALERT — Possible fall detected!'
                '</div>',
                unsafe_allow_html=True
            )


    live_dashboard()


# ============================================================
# UPLOAD VIDEO
# ============================================================
else:

    st.subheader(
        "📹 Upload Video Analysis"
    )


    uploaded = st.file_uploader(
        "Upload crowd video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded:

        import tempfile


        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )


        temp.write(
            uploaded.read()
        )

        temp.close()


        cap = cv2.VideoCapture(
            temp.name
        )


        video_box = st.empty()


        history = defaultdict(list)


        while cap.isOpened():

            ret, frame = cap.read()


            if not ret:

                break


            annotated, people = process_frame(
                frame,
                confidence,
                history,
                show_ids
            )


            counts = Counter(
                person["action"]
                for person in people
            )


            data = {

                "People":
                    len(people),

                "Walking":
                    counts["Walking"],

                "Running":
                    counts["Running"],

                "Standing":
                    counts["Standing"],

                "Sitting":
                    counts["Sitting"],

                "Hand Raised":
                    counts["Hand Raised"],

                "Fall":
                    counts["Fall"]
            }


            update_dashboard(data)


            rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )


            video_box.image(
                rgb,
                channels="RGB"
            )


        cap.release()


        st.success(
            "✅ Video analysis completed."
        )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "CrowdVision AI • YOLO Pose • "
    "Multi-Person Tracking • Action Recognition"
)