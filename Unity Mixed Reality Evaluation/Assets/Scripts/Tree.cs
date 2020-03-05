using Vuforia;
using UnityEngine;
using System; 
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;


public class Tree : MonoBehaviour
{
    public double distance;
    public double mu_l;      // fingertip distance lower limit
    public double mu_u;      // fingertip distance upper limit  
    public double lambda_l;  // object scale lower limit 
    public double lambda_u;  // object scale upper limit 

    void Start ()
    {   
        mu_l = 100; 
        mu_u = 180;
        lambda_l = 0.04; 
        lambda_u = 0.16;  
    }

    void Update ()
    {
        // Distance
        if (distance > mu_u)
        {
            transform.localScale = new Vector3((float)lambda_u, (float)lambda_u, (float)lambda_u);
        }
        else if (distance <= mu_u && distance >=mu_l)
        {
            double slope = (lambda_u - lambda_l) / (mu_u - mu_l);
            double intercept = (lambda_l * mu_u - lambda_u * mu_l) / (mu_u - mu_l);  
            double scale = slope * distance + intercept;
            transform.localScale = new Vector3((float)scale, (float)scale, (float)scale);
        }
        else if (distance < mu_l)
        {
            transform.localScale = new Vector3((float)lambda_l, (float)lambda_l, (float)lambda_l);
        }

        try
        {
            using (var tcp = new TcpClient("127.0.0.1", 8888))
            {
                byte[] message = new byte[1024];
                distance = tcp.GetStream().Read(message, 0, message.Length);
                Debug.Log("Received distance: " + distance.ToString());
            }
        }
        catch
        {
            UnityEngine.Debug.Log("No data received");
        }
    }
}
