using Vuforia;
using System;
using System.Net;
using System.Text;
using UnityEngine;
using System.Net.Sockets;
using System.Collections;
using System.Threading.Tasks;
using System.Collections.Generic;


public class Helicopter : MonoBehaviour
{
    public double dist;                 // amount of distance 
    public double angle;                // amount of angle 
    public double ty;                   // translation along y axis 
    public double tz;                   // translation along z axis 

    public double fingertip_l;          // fingertip distance lower limit
    public double fingertip_u;          // fingertip distance upper limit  

    public double scale_l;              // object scale lower limit 
    public double scale_u;              // object scale upper limit 
 
    public double translate_l_y;        // object translation lower limit along y 
    public double translate_u_y;        // object translation upper limit along y 
    public double translate_l_z;        // object translation lower limit along z
    public double translate_u_z;        // object translation upper limit along z

    public double slope, intercept, scale;
    public double min_finger_pos, max_finger_pos, pos_y, pos_z; 

    void Start ()
    {
        // initializing 
        dist = 0;
        angle = 0;
        ty = 0;
        tz = 0; 

        // defining threshold values 
        fingertip_l = 100.0; 
        fingertip_u = 180.0;
        max_finger_pos = 1;
        min_finger_pos = 0;

        scale_l = 0.05; 
        scale_u = 0.20;

        translate_l_y = -0.5;
        translate_u_y = 0.5; 
        translate_l_z = -1.0;
        translate_u_z = 1.0;
    }

    void Update ()
    {
        // scale transformation
        if (dist > fingertip_u)
        {
            transform.localScale = new Vector3((float)scale_u, (float)scale_u, (float)scale_u);
        }

        else if (dist <= fingertip_u && dist >= fingertip_l)
        {
            slope = (scale_u - scale_l) / (fingertip_u - fingertip_l);
            intercept = scale_l - slope * fingertip_l;
            scale = slope * dist + intercept;
            transform.localScale = new Vector3((float)scale, (float)scale, (float)scale);
        }

        else if (dist < fingertip_l)
        {
            transform.localScale = new Vector3((float)scale_l, (float)scale_l, (float)scale_l);
        }

        // rotation transformation
        transform.localRotation = Quaternion.Euler((float)angle, 0.0f, 0.0f);

        // translation transformation
        slope = (translate_u_y - translate_l_y) / (max_finger_pos - min_finger_pos);
        intercept = translate_l_y - slope * min_finger_pos;
        pos_y = slope * tz + intercept;

        slope = (translate_u_z - translate_l_z) / (max_finger_pos - min_finger_pos);
        intercept = translate_l_z - slope * min_finger_pos;
        pos_z = slope * ty + intercept;

        transform.localPosition = new Vector3(0.0f, (float)pos_y, (float)pos_z);

        try
        {
            using (var tcp = new TcpClient("127.0.0.1", 8888))
            {
                byte[] message = new byte[1024];
                int len = tcp.GetStream().Read(message, 0, message.Length);
                string str = Encoding.ASCII.GetString(message, 0, len);
                string[] str_list = str.Split(',');
                dist = Convert.ToDouble(str_list[0]);
                angle = Convert.ToDouble(str_list[1]);
                ty = Convert.ToDouble(str_list[2]);
                tz = Convert.ToDouble(str_list[3]);
                Debug.Log("Received: " + dist + " " + angle + " " + ty + ", " + tz);
            }
        }
        catch
        {
            UnityEngine.Debug.Log("No data received");
        }

    }
}
