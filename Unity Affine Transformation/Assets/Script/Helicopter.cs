using Vuforia;
using UnityEngine;
using System; 
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;


public class Helicopter : MonoBehaviour
{
    public double value;                // type of transformation 
    public double data;                 // amount of transformation

    public double fingertip_l;          // fingertip distance lower limit
    public double fingertip_u;          // fingertip distance upper limit  
    public double scale_l;              // object scale lower limit 
    public double scale_u;              // object scale upper limit 
    public double rotate_l;             // object rotation lower limit 
    public double rotate_u;             // object rotation upper limit 
    public double translate_l;          // object translation lower limit 
    public double translate_u;          // object translation upper limit 

    public double slope, intercept;
    public double scale, angle, position; 

    void Start ()
    {
        // initializing 
        value = 0;
        data = 0;

        // defining threshold values 
        fingertip_l = 100.0; 
        fingertip_u = 180.0;
        scale_l = 0.05; 
        scale_u = 0.2;
        rotate_l = 0.0;
        rotate_u = 45.0;
        translate_l = 0.0;
        translate_u = 0.5;
    }

    void Update ()
    {
        // unified
        if (value == 0)
        {
            if (data > fingertip_u)
            {
                transform.localScale = new Vector3((float)scale_u, (float)scale_u, (float)scale_u);
                transform.localRotation = Quaternion.Euler((float)rotate_u, 0.0f, 0.0f);
                transform.localPosition = new Vector3(0.0f, (float)translate_u, 0.0f);
            }

            else if (data <= fingertip_u && data >= fingertip_l)
            {
                slope = (scale_u - scale_l) / (fingertip_u - fingertip_l);
                intercept = scale_l - slope * fingertip_l; 
                scale = slope * data + intercept;
                transform.localScale = new Vector3((float)scale, (float)scale, (float)scale);

                slope = (rotate_u - rotate_l) / (fingertip_u - fingertip_l);
                intercept = rotate_l - slope * fingertip_l;
                angle = slope * data + intercept;
                transform.localRotation = Quaternion.Euler((float)angle, 0.0f, 0.0f);

                slope = (translate_u - translate_l) / (fingertip_u - fingertip_l);
                intercept = translate_l - slope * fingertip_l;
                position = slope * data + intercept;
                transform.localPosition = new Vector3(0.0f, (float)position, 0.0f);
            }

            else if (data < fingertip_l)
            {
                transform.localScale = new Vector3((float)scale_l, (float)scale_l, (float)scale_l);
                transform.localRotation = Quaternion.Euler(0.0f, 0.0f, 0.0f);
                transform.localPosition = new Vector3(0.0f, 0.0f, 0f);
            }
        }

        // scale transformation
        if (value == 1)
        {
            if (data > fingertip_u)
            {
                transform.localScale = new Vector3((float)scale_u, (float)scale_u, (float)scale_u);
            }

            else if (data <= fingertip_u && data >= fingertip_l)
            {
                slope = (scale_u - scale_l) / (fingertip_u - fingertip_l);
                intercept = scale_l - slope * fingertip_l;
                scale = slope * data + intercept;
                transform.localScale = new Vector3((float)scale, (float)scale, (float)scale);
            }

            else if (data < fingertip_l)
            {
                transform.localScale = new Vector3((float)scale_l, (float)scale_l, (float)scale_l);
            }
        }

        // rotation transformation
        if (value == 2)
        {
            if (data > fingertip_u)
            {
                transform.localRotation = Quaternion.Euler((float)rotate_u, 0.0f, 0.0f);
            }

            else if (data <= fingertip_u && data >= fingertip_l)
            {
                slope = (rotate_u - rotate_l) / (fingertip_u - fingertip_l);
                intercept = rotate_l - slope * fingertip_l;
                angle = slope * data + intercept;
                transform.localRotation = Quaternion.Euler((float)angle, 0.0f, 0.0f);
            }

            else if (data < fingertip_l)
            {
                transform.localRotation = Quaternion.Euler(0.0f, 0.0f, 0.0f);
            }
        }

        // translation transformation 
        if (value == 3)
        {
            if (data > fingertip_u)
            {
                transform.localPosition = new Vector3(0.0f, (float)translate_u, 0.0f);
            }

            else if (data <= fingertip_u && data >= fingertip_l)
            {
                slope = (translate_u - translate_l) / (fingertip_u - fingertip_l);
                intercept = translate_l - slope * fingertip_l;
                position = slope * data + intercept;
                transform.localPosition = new Vector3(0.0f, (float)position, 0.0f);
            }

            else if (data < fingertip_l)
            {
                transform.localPosition = new Vector3(0.0f, 0.0f, 0f);
            }
        }

        try
        {
            using (var tcp = new TcpClient("127.0.0.1", 8888))
            {
                byte[] message = new byte[1024];
                int len = tcp.GetStream().Read(message, 0, message.Length);
                string str = Encoding.ASCII.GetString(message, 0, len);
                string[] str_list = str.Split(',');
                value = Convert.ToDouble(str_list[0]);
                data = Convert.ToDouble(str_list[1]);
                Debug.Log("Received: " + value + " " + data);
            }
        }
        catch
        {
            UnityEngine.Debug.Log("No data received");
        }

    }
}
