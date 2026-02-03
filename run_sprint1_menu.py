import os
import sys
import subprocess
import time

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_separator():
    print("-" * 75)

def print_header():
    print("=" * 75)
    print("SPRINT 1: GEOMETRY OF FORMATION - PIPELINE MANAGER")
    print("=" * 75)
    print(f"Working Directory: {os.getcwd()}")
    print_separator()

def run_script(script_name, verification_text, automated_mode=False):
    """
    Executes a script and prints the 'Verification' instructions first.
    """
    if not os.path.exists(script_name):
        print(f"\n[ERROR] File '{script_name}' not found.")
        if not automated_mode:
            input("\nPress Enter to return to the menu...")
        return

    # --- VERIFICATION BLOCK ---
    print(f"\n[LAUNCH] Starting: {script_name}")
    print("=" * 75)
    print(f"[VERIFY] WHAT TO EXPECT / CHECK:")
    print(f"   {verification_text}")
    print("=" * 75)
    
    if not automated_mode:
        input("Press Enter to run...")
    else:
        print("Running in 2 seconds...")
        time.sleep(2)
    
    # Run the script
    start_time = time.time()
    try:
        subprocess.run([sys.executable, script_name], check=False)
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user.")
        if automated_mode: return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    elapsed = time.time() - start_time
    print("-" * 75)
    print(f"[STATUS] Finished in {elapsed:.2f} seconds.")
    
    if not automated_mode:
        input("Press Enter to return to the menu...")
    else:
        print("Next step in 3 seconds... (Ctrl+C to stop)")
        time.sleep(3)
    
    return True

def main():
    # Format: (Title, Filename, Description, VERIFICATION_INSTRUCTION)
    scripts = [
        (
            "Step 1: Camera Calibration", 
            "calibrate_camera.py", 
            "Calculates lens distortion from checkerboard images.",
            "Look for 'calibration successful' message. Open 'verification_2_undistorted.jpg' and check if the checkerboard edges are perfectly straight (not curved)."
        ),
        (
            "Step 2: Calculate Homography", 
            "calculate_homography.py", 
            "Calculates the top-down perspective transform.",
            "Open 'verification_3_birdseye.jpg'. The checkerboard must be a PERFECT RECTANGLE. If it looks like a trapezoid, the calibration failed."
        ),
        (
            "Step 3: Fix Resolution Mismatch", 
            "fix_resolution.py", 
            "Rescales math if video size != photo size.",
            "Expect a message: 'Success! Created geometry_pipeline_video.pkl'. If it says 'Resolutions match', that is also fine."
        ),
        (
            "Step 4: Test on Video (Basic)", 
            "test_on_video.py", 
            "Quick test with standard 1000px width.",
            "A window opens. The road will look NARROW and ZOOMED IN. This is normal behavior for this test. Just check if lines are parallel."
        ),
        (
            "Step 5: Test on Video (Wide Canvas)", 
            "test_on_video_wide.py", 
            "High-res test with shift controls.",
            "A window opens. Is the road centered? If it is cut off, open the script and adjust 'SHIFT_X'. Ensure you see the full lane width."
        ),
        (
            "Step 6: Create Demo Reel", 
            "create_side_by_side.py", 
            "Exports split-screen video.",
            "Wait for processing to finish. Open 'sprint1_demo_reel.mp4'. As the car moves, road features should hit the bottom of both screens at the same time."
        ),
        (
            "Step 7: RUN FINAL PIPELINE", 
            "pipeline_sprint1_formation.py", 
            "Generates final MP4 and frame images.",
            "This will take time. Check the 'sprint1_frames' folder. Open a few JPGs and confirm they are high-quality top-down maps."
        )
    ]

    while True:
        clear_terminal()
        print_header()
        print("Select an operation to execute:\n")
        
        for i, (title, filename, desc, verify) in enumerate(scripts):
            print(f" {i+1}. {title}")
            print(f"    Info: {desc}")
            print("") 
        
        print("-" * 30)
        print(" 9. RUN ALL (Execute Steps 1-7 Sequentially)")
        print("-" * 30)
        print(" Q. Quit")
        print_separator()
        
        choice = input("Enter selection: ").strip().lower()

        if choice == 'q':
            break
        
        if choice == '9':
            print("\n[START] STARTING FULL SEQUENCE...")
            time.sleep(1)
            for title, filename, desc, verify in scripts:
                if not run_script(filename, verify, automated_mode=True): break
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scripts):
                run_script(scripts[idx][1], scripts[idx][3])
            else:
                print("Invalid number.")
                time.sleep(1)

if __name__ == "__main__":
    main()