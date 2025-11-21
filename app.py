import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import io
import tempfile
import os
import google.generativeai as genai
from datetime import datetime
import json
from dotenv import load_dotenv
from prompts import get_frame_analysis_prompt, get_final_summary_prompt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import subprocess
import shutil
import glob
import atexit

# NEW: imageio-ffmpeg provides a bundled ffmpeg binary usable in no-root envs (Streamlit Cloud)
try:
    import imageio_ffmpeg as iio_ffmpeg
    FFMPEG_EXE = iio_ffmpeg.get_ffmpeg_exe()
except Exception:
    iio_ffmpeg = None
    FFMPEG_EXE = None

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')

# Create required directories
LOGS_DIR = "logs"
COMPRESSED_DIR = "compressed"
FRAMES_DIR = "frames"
ANNOTATED_DIR = "annotated"

for directory in [LOGS_DIR, COMPRESSED_DIR, FRAMES_DIR, ANNOTATED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_data = joblib.load('jsk_csk_2025.joblib')
    return model_data['model'], model_data['label_encoder']

clf, label_encoder = load_model()

# Initialize ArcFace model
@st.cache_resource
def load_face_analysis():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

face_app = load_face_analysis()

# Streamlit UI Configuration
st.set_page_config(page_title="Cricket Player Face Recognition", layout="wide")
st.title("🏏 Cricket Player Face Recognition with AI Analysis")
st.write("Upload images or videos to recognize cricket players with AI-powered frame analysis.")

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0.0, 100.0, 99.99, 0.01)
sharpness_threshold = st.sidebar.slider("Sharpness Threshold", 0, 100, 10, 1)
show_bounding_box = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Score", value=True)

# Video specific settings
st.sidebar.header("🎥 Video Settings (Ultra Fast)")
frames_per_second = st.sidebar.slider("Extract Frames Per Second", 0.1, 5.0, 1.0, 0.1,
                                     help="How many frames to extract per second of video")
enable_ai_analysis = st.sidebar.checkbox("Enable AI Frame Analysis (Gemini)", value=True,
                                        help="Analyze each frame with detected players using Gemini AI")
parallel_workers = st.sidebar.slider("Parallel Workers (LLM)", 1, 8, 4, 1,
                                    help="Number of parallel workers for LLM analysis")
skip_compression = st.sidebar.checkbox("Skip Compression (for small files)", value=True,
                                       help="Skip compression step for faster processing")

# Upload mode selection
upload_mode = st.radio("Select Upload Mode:", ["Images", "Video"], horizontal=True)

# ================== UTILITY FUNCTIONS ==================

def save_log(log_data, log_type="video_analysis"):
    """Save logs to JSON file - handles numpy types"""
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "image"}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    log_data = clean(log_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{log_type}_{timestamp}.json")
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    return log_file

def compute_sharpness(face_img):
    """Compute Laplacian variance for sharpness"""
    if face_img is None or face_img.size == 0:
        return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_video_duration(video_path):
    """Try to get duration by parsing ffmpeg stderr output"""
    exe = FFMPEG_EXE or shutil.which('ffmpeg')
    if not exe:
        return 0
    try:
        cmd = [exe, '-i', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        stderr = result.stderr or result.stdout or ""
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
        if m:
            hours = int(m.group(1))
            minutes = int(m.group(2))
            seconds = float(m.group(3))
            return hours * 3600 + minutes * 60 + seconds
        return 0
    except Exception:
        return 0

def clear_folder(folder_path):
    """Clear all files in a folder"""
    for file in glob.glob(os.path.join(folder_path, "*")):
        try:
            os.remove(file)
        except:
            pass

def cleanup_processing_folders():
    """Clean up frames and annotated folders after processing"""
    try:
        clear_folder(FRAMES_DIR)
        clear_folder(ANNOTATED_DIR)
        st.info("🧹 Temporary files cleaned up successfully")
    except Exception as e:
        st.warning(f"⚠️ Cleanup warning: {str(e)}")

# ================== OPTIMIZED FFMPEG EXTRACTION ==================

def extract_frames_ffmpeg(video_path, output_folder, fps_extract=1.0):
    """ULTRA-FAST frame extraction with optimizations"""
    exe = FFMPEG_EXE or shutil.which('ffmpeg')
    if not exe:
        st.error("❌ FFmpeg not found! Add 'imageio-ffmpeg' to your requirements or install ffmpeg system-wide.")
        return []
    
    clear_folder(output_folder)
    
    duration = get_video_duration(video_path)
    expected_frames = int(duration * fps_extract) if duration > 0 else "unknown"
    st.info(f"🎯 Extracting ~{expected_frames} frames ({fps_extract} fps)")
    
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    
    command = [
        exe,
        '-skip_frame', 'nokey',
        '-i', video_path,
        '-vf', f'fps={fps_extract}',
        '-q:v', '10',
        '-threads', '0',
        '-preset', 'ultrafast',
        '-y',
        output_pattern
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            st.error(f"❌ FFmpeg error: {result.stderr[:200]}")
            return []
        
        frame_files = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))
        
        if not frame_files:
            st.error("❌ No frames extracted")
            return []
        
        frame_paths = []
        for idx, frame_path in enumerate(frame_files):
            timestamp_sec = idx / fps_extract if fps_extract > 0 else idx
            frame_paths.append({
                'path': frame_path,
                'second': int(timestamp_sec),
                'frame_num': int(idx)
            })
        
        st.success(f"✅ Extracted {len(frame_paths)} frames")
        return frame_paths
        
    except subprocess.TimeoutExpired:
        st.error("❌ FFmpeg timed out")
        return []
    except Exception as e:
        st.error(f"❌ FFmpeg error: {str(e)}")
        return []

# ================== OPTIMIZED FACE DETECTION ==================

def detect_faces_sequential(frame_paths, conf_threshold, sharp_threshold):
    """OPTIMIZED: Process frames with early stopping and reduced overhead"""
    results = []
    frames_with_players = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = len(frame_paths)
    
    for idx, frame_info in enumerate(frame_paths):
        if idx % 5 == 0 or idx == total_frames - 1:
            progress_bar.progress((idx + 1) / total_frames)
            status_text.text(f"🔍 Detecting: {idx + 1}/{total_frames} frames")
        
        try:
            frame = cv2.imread(frame_info['path'], cv2.IMREAD_COLOR)
            if frame is None:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_app.get(frame_rgb)
            
            if not faces:
                results.append({
                    'frame_info': frame_info,
                    'detections': [],
                    'annotated_path': None
                })
                continue
            
            detections = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                face_crop = frame_rgb[y1:y2, x1:x2]
                sharpness = compute_sharpness(face_crop)
                if sharpness < sharp_threshold:
                    continue
                
                embedding = face.embedding
                predicted_label = clf.predict([embedding])[0]
                predicted_name = label_encoder.inverse_transform([predicted_label])[0]
                
                confidence = None
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba([embedding])
                    confidence = np.max(proba) * 100
                
                if confidence is not None and confidence < conf_threshold:
                    continue
                
                detections.append({
                    'name': predicted_name,
                    'confidence': float(confidence) if confidence else 0.0,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
            
            annotated_path = None
            if detections:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{det['name']} {det['confidence']:.1f}%", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                annotated_path = os.path.join(ANNOTATED_DIR, f"annotated_{frame_info['second']}.jpg")
                cv2.imwrite(annotated_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            result = {
                'frame_info': frame_info,
                'detections': detections,
                'annotated_path': annotated_path
            }
            results.append(result)
            
            if detections:
                frames_with_players.append(result)
                
        except Exception as e:
            st.warning(f"⚠️ Frame {idx} error: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Detection: {len(frames_with_players)}/{total_frames} frames")
    
    return results, frames_with_players

# ================== PARALLEL LLM ANALYSIS ==================

def analyze_frame_llm_worker(args):
    """Worker for parallel LLM analysis"""
    frame_data = args
    
    try:
        if not frame_data['detections']:
            return None
        
        image = Image.open(frame_data['annotated_path'])
        players = [d['name'] for d in frame_data['detections']]
        prompt = get_frame_analysis_prompt(players)
        
        response = model.generate_content([prompt, image])
        
        return {
            'frame_info': frame_data['frame_info'],
            'players': players,
            'description': response.text if response and response.text else "No response"
        }
    
    except Exception as e:
        return {
            'frame_info': frame_data['frame_info'],
            'players': [d['name'] for d in frame_data['detections']],
            'description': f"Error: {str(e)}"
        }

def parallel_llm_analysis(frames_with_players, num_workers=4):
    """OPTIMIZED: Process in batches to reduce API overhead"""
    if not frames_with_players:
        return []
    
    analyses = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_frame_llm_worker, frame) 
                   for frame in frames_with_players]
        
        completed = 0
        total = len(futures)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for future in as_completed(futures):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"🤖 AI: {completed}/{total} frames")
            
            try:
                result = future.result(timeout=30)
                if result:
                    analyses.append(result)
            except Exception as e:
                st.warning(f"⚠️ LLM error: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
    
    st.success(f"✅ AI: {len(analyses)} descriptions")
    
    return analyses

# ================== FINAL SUMMARY GENERATION ==================

def generate_final_summary(llm_analyses, all_detected_players):
    """Generate final summary using existing prompt from prompts.py"""
    if not GEMINI_API_KEY or not llm_analyses:
        return None
    
    try:
        frame_analyses = []
        for analysis in llm_analyses:
            frame_info = analysis.get('frame_info', {})
            frame_analyses.append({
                'frame': frame_info.get('frame_num', 0),
                'timestamp': float(frame_info.get('second', 0)),
                'players': analysis.get('players', []),
                'analysis': analysis.get('description', '')
            })
        
        prompt = get_final_summary_prompt(frame_analyses, all_detected_players)
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        return None
        
    except Exception as e:
        st.error(f"❌ Summary error: {str(e)}")
        return None

# ================== COMPLETE ULTRA-FAST PIPELINE ==================

def process_video_ultrafast(video_path, conf_threshold, sharp_threshold, fps_extract, num_workers):
    """Ultra-fast 3-step pipeline + Final Summary"""
    pipeline_start = datetime.now()
    
    # Step 1: FFmpeg Direct Frame Extraction
    st.markdown("### 🎞️ Step 1: FFmpeg Frame Extraction")
    step1_start = datetime.now()
    frame_paths = extract_frames_ffmpeg(video_path, FRAMES_DIR, fps_extract)
    step1_time = (datetime.now() - step1_start).total_seconds()
    st.info(f"⏱️ Extraction time: {step1_time:.2f}s")
    
    if not frame_paths:
        st.error("❌ No frames extracted")
        return None
    
    # Step 2: Sequential Face Detection
    st.markdown("### 👤 Step 2: Face Detection (Cached Model)")
    step2_start = datetime.now()
    all_results, frames_with_players = detect_faces_sequential(frame_paths, conf_threshold, sharp_threshold)
    step2_time = (datetime.now() - step2_start).total_seconds()
    st.info(f"⏱️ Detection time: {step2_time:.2f}s")
    
    # Step 3: Parallel LLM Analysis
    step3_time = 0
    llm_analyses = []
    if enable_ai_analysis and GEMINI_API_KEY and frames_with_players:
        st.markdown("### 🤖 Step 3: Parallel AI Analysis")
        step3_start = datetime.now()
        llm_analyses = parallel_llm_analysis(frames_with_players, num_workers)
        step3_time = (datetime.now() - step3_start).total_seconds()
        st.info(f"⏱️ LLM analysis time: {step3_time:.2f}s ({num_workers} workers)")
    
    # Collect all detected players
    all_players = set()
    for result in all_results:
        for detection in result['detections']:
            all_players.add(detection['name'])
    
    # Step 4: Generate Final Summary
    step4_time = 0
    final_summary = None
    if enable_ai_analysis and GEMINI_API_KEY and llm_analyses:
        st.markdown("### 📊 Step 4: Generating Final Summary")
        step4_start = datetime.now()
        with st.spinner("🧠 Generating comprehensive summary..."):
            final_summary = generate_final_summary(llm_analyses, list(all_players))
        step4_time = (datetime.now() - step4_start).total_seconds()
        st.info(f"⏱️ Summary time: {step4_time:.2f}s")
    
    total_time = (datetime.now() - pipeline_start).total_seconds()
    
    return {
        'all_results': all_results,
        'frames_with_players': frames_with_players,
        'llm_analyses': llm_analyses,
        'final_summary': final_summary,
        'all_detected_players': list(all_players),
        'timing': {
            'extraction': float(step1_time),
            'detection': float(step2_time),
            'llm': float(step3_time),
            'summary': float(step4_time),
            'total': float(total_time)
        }
    }

# ================== IMAGE MODE ==================

def process_image(image_np, conf_threshold, sharp_threshold, show_bbox, show_conf):
    """Process single image"""
    image_rgb = image_np.copy()
    faces = face_app.get(image_rgb)

    if not faces:
        return image_rgb, None, "No face detected"

    results = []
    
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_crop = image_rgb[y1:y2, x1:x2]

        sharpness = compute_sharpness(face_crop)
        if sharpness < sharp_threshold:
            continue

        embedding = face.embedding
        predicted_label = clf.predict([embedding])[0]
        predicted_name = label_encoder.inverse_transform([predicted_label])[0]

        confidence = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba([embedding])
            confidence = np.max(proba) * 100

        if confidence is not None and confidence < conf_threshold:
            continue

        results.append((x1, y1, x2, y2, predicted_name, confidence))

    if not results:
        return image_rgb, None, "No high-confidence face detected"

    final_results = {}
    for x1, y1, x2, y2, name, confidence in results:
        if name not in final_results or final_results[name][1] < confidence:
            final_results[name] = ((x1, y1, x2, y2), confidence)

    sorted_results = sorted(final_results.items(), key=lambda x: x[1][1], reverse=True)

    if show_bbox:
        for name, ((x1, y1, x2, y2), confidence) in sorted_results:
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font_scale = max(1.5, (x2 - x1) / 100)
            thickness = max(2, int(font_scale * 3))
            
            if show_conf:
                label_text = f"{name} ({confidence:.2f}%)"
            else:
                label_text = name
                
            cv2.putText(image_rgb, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

    return image_rgb, sorted_results, None

# ================== MAIN APPLICATION ==================

if upload_mode == "Images":
    st.header("📸 Image Recognition")
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], 
                                     accept_multiple_files=True)
    
    csv_results = []
    
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                image = Image.open(uploaded_file)
                image_rgb = np.array(image.convert('RGB'))
                
                if image_rgb is None or image_rgb.size == 0:
                    raise ValueError(f"Uploaded image {uploaded_file.name} is empty or could not be loaded.")
                
                processed_image, results, message = process_image(image_rgb.copy(), confidence_threshold,
                                                          sharpness_threshold, show_bounding_box, 
                                                          show_confidence)
                
                processed_pil = Image.fromarray(processed_image)
                
                if results:
                    caption_text = "Results: "
                    for name, ((x1, y1, x2, y2), confidence) in results:
                        csv_results.append([i, uploaded_file.name, name, confidence])
                        caption_text += f"{name} ({confidence:.2f}%), "
                    caption_text = caption_text.rstrip(", ")
                    st.image(processed_pil, caption=caption_text, use_container_width=True)
                else:
                    csv_results.append([i, uploaded_file.name, "No face detected", None])
                    st.image(processed_pil, caption=message, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if csv_results:
            df = pd.DataFrame(csv_results, columns=["S.No", "Image Path", "Player Name", "Confidence"])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode()
            st.download_button(label="📥 Download Results as CSV", data=csv_bytes, 
                             file_name="face_recognition_results.csv", mime="text/csv")

else:  # Video mode
    st.header("🎥 Video Recognition - Ultra Fast Mode")
    
    if not GEMINI_API_KEY:
        st.warning("⚠️ GEMINI_API_KEY not found in .env file. AI analysis will be disabled.")
    
    exe = FFMPEG_EXE or shutil.which('ffmpeg')
    if not exe:
        st.error("❌ FFmpeg not found! For Streamlit Cloud add 'imageio-ffmpeg' to your requirements.txt.")
        st.markdown("**Install FFmpeg in this environment:**")
        st.code("Add to requirements.txt:\nimageio-ffmpeg\n\nOr install ffmpeg system-wide for self-hosted deployments.")
    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "flv", "wmv"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        # tfile.write(uploaded_video.read())
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        CHUNK_SIZE = 1024 * 1024  # 1 MB
        
        video_bytes = uploaded_video.stream  # raw binary stream
        
        while True:
            chunk = video_bytes.read(CHUNK_SIZE)
            if not chunk:
                break
            tfile.write(chunk)
        
        tfile.close()
        
        st.video(uploaded_video)
        
        if st.button("🚀 Process Video (Ultra Fast)", type="primary"):
            
            # Clear previous results
            clear_folder(FRAMES_DIR)
            clear_folder(ANNOTATED_DIR)
            
            with st.spinner("🚀 Processing with ultra-fast pipeline..."):
                video_data = process_video_ultrafast(
                    tfile.name, 
                    confidence_threshold, 
                    sharpness_threshold,
                    frames_per_second,
                    parallel_workers
                )
            
            if video_data:
                st.success("✅ Video processing complete!")
                
                # Display timing stats
                st.subheader("⏱️ Performance Metrics")
                timing = video_data['timing']
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Extraction", f"{timing['extraction']:.1f}s")
                with col2:
                    st.metric("Detection", f"{timing['detection']:.1f}s")
                with col3:
                    st.metric("LLM Analysis", f"{timing['llm']:.1f}s")
                with col4:
                    st.metric("Summary", f"{timing.get('summary', 0):.1f}s")
                with col5:
                    st.metric("Total Time", f"{timing['total']:.1f}s", delta=f"{len(video_data['frames_with_players'])} frames")
                
                # Display detected players
                st.subheader("🎯 Detected Players")
                if video_data['all_detected_players']:
                    st.success(f"✅ Found {len(video_data['all_detected_players'])} unique player(s): {', '.join(video_data['all_detected_players'])}")
                else:
                    st.warning("⚠️ No players detected in the video")
                
                # Display final summary
                if video_data.get('final_summary'):
                    st.subheader("📊 Final AI Summary")
                    st.markdown("---")
                    st.markdown(video_data['final_summary'])
                    st.markdown("---")
                
                # Display frame results
                if video_data['frames_with_players']:
                    st.subheader("📋 Detection Results")
                    for result in video_data['frames_with_players']:
                        frame_info = result['frame_info']
                        detections = result['detections']
                        
                        with st.expander(f"Frame @ {frame_info['second']}s - {len(detections)} player(s)"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if result['annotated_path'] and os.path.exists(result['annotated_path']):
                                    st.image(result['annotated_path'], caption=f"Frame {frame_info['second']}")
                            with col2:
                                for det in detections:
                                    st.write(f"**{det['name']}** - {det['confidence']:.2f}%")
                
                # Display AI analyses
                if video_data['llm_analyses']:
                    st.subheader("🤖 AI Frame Analyses")
                    with st.expander("📋 View All Frame Analyses", expanded=False):
                        for analysis in video_data['llm_analyses']:
                            frame_info = analysis['frame_info']
                            players = ", ".join(analysis['players'])
                            
                            st.markdown(f"**Frame @ {frame_info['second']}s** - Players: *{players}*")
                            st.info(analysis['description'])
                            st.markdown("---")
                
                # Save logs
                log_file = save_log(video_data, "video_analysis")
                st.success(f"✅ Analysis saved to: {log_file}")
                
                # Download log
                with open(log_file, 'r') as f:
                    log_json = f.read()
                st.download_button(
                    label="📥 Download Full Analysis Log (JSON)",
                    data=log_json,
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # ========== AUTO CLEANUP AFTER OUTPUT ==========
                cleanup_processing_folders()
            
            # Cleanup temp file
            os.unlink(tfile.name)
