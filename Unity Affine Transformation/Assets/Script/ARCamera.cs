using Vuforia;
using UnityEngine;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;


public class ARCamera : MonoBehaviour
{
    public int frameRate = 20;
    public int sizeMultiplier = 1;
    public int counter = 1;

    static int port = 8888;
    static string ip_adress = "127.0.0.1";
    Socket client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
    IPEndPoint ep = new IPEndPoint(IPAddress.Parse(ip_adress), port);

    private bool mFormatRegistered = false;
    private Image.PIXEL_FORMAT mPixelFormat = Image.PIXEL_FORMAT.RGBA8888;  // GRAYSCALE;

    void Start ()
    {
        Time.captureFramerate = frameRate;

        try
        {
            client.Connect(ep);
        }
        catch { }
        VuforiaARController.Instance.RegisterVuforiaStartedCallback(OnVuforiaStarted);
    }
	
	void Update ()
    {
        if (mFormatRegistered)
        {
            Vuforia.Image image = CameraDevice.Instance.GetCameraImage(mPixelFormat);
            if (image != null)
            {
                // Saving screenshots
                // var name = string.Format("{0}/image{1:D04}.jpg", "Screenshots", Time.frameCount);
                // ScreenCapture.CaptureScreenshot(name, sizeMultiplier);

                byte[] pixels = image.Pixels;
                try
                {
                    client.Send(pixels, 0, pixels.Length, SocketFlags.None);
                    UnityEngine.Debug.Log("ARCamera: Image data transmitted.");
                }
                catch
                {
                    UnityEngine.Debug.Log("No data transmitted.");
                }
            }
        }
    }

    void OnVuforiaStarted()
    {
        if (CameraDevice.Instance.SetFrameFormat(mPixelFormat, true))
        {
            UnityEngine.Debug.Log("Successfully registered pixel format " + mPixelFormat.ToString());

            mFormatRegistered = true;
        }
        else
        {
            UnityEngine.Debug.LogError(
                "\nFailed to register pixel format: " + mPixelFormat.ToString() +
                "\nThe format may be unsupported by your device." +
                "\nConsider using a different pixel format.\n");

            mFormatRegistered = false;
        }
    }
}
