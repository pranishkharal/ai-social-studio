import os
import sys

def create_shortcut():
    desktop = os.path.join(os.environ.get("USERPROFILE", "C:/Users/MSI"), "Desktop")
    shortcut_path = os.path.join(desktop, "AI Social Studio.lnk")
    target_path = "D:\\kick-ai-clip-engine\\start.bat"
    working_dir = "D:\\kick-ai-clip-engine"
    icon_path = "D:\\kick-ai-clip-engine\\static\\app_icon.ico"

    vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{target_path}"
oLink.WorkingDirectory = "{working_dir}"
oLink.Description = "AI Social Studio v1.0 - Commercial Kick Live Stream Clipping"
oLink.IconLocation = "{icon_path},0"
oLink.Save
"""
    vbs_file = "D:\\kick-ai-clip-engine\\temp_shortcut.vbs"
    with open(vbs_file, "w") as f:
        f.write(vbs_script)

    os.system(f'cscript //nologo "{vbs_file}"')
    if os.path.exists(vbs_file):
        os.remove(vbs_file)

    if os.path.exists(shortcut_path):
        print(f"SUCCESS: Desktop shortcut created at: {shortcut_path}")
        return True
    else:
        print("FAILED to create shortcut.")
        return False

if __name__ == "__main__":
    create_shortcut()
