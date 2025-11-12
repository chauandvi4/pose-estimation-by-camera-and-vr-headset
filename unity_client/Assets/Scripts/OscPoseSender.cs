using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

/// <summary>
/// Collects Quest HMD/upper-body joint poses and streams them as a JSON payload
/// wrapped inside an OSC message over UDP.
/// </summary>
public class OscPoseSender : MonoBehaviour
{
    [Serializable]
    public class JointBinding
    {
        [Tooltip("Human-readable name that will be used inside the OSC payload (e.g. chest, leftShoulder).")]
        public string jointName;

        [Tooltip("Transform providing the pose for the joint.")]
        public Transform transform;

        [Tooltip("Send this joint even if the transform is currently disabled.")]
        public bool forceSend;

        public bool IsValid => transform != null && (transform.gameObject.activeInHierarchy || forceSend);
    }

    [Serializable]
    public class PoseSample
    {
        public Vector3 position;
        public Quaternion rotation;
    }

    [Serializable]
    public class JointSample
    {
        public string name;
        public PoseSample pose;
    }

    [Serializable]
    public class PosePacket
    {
        public double timestamp;
        public string source = "meta_quest";
        public string space = "world";
        public PoseSample hmd;
        public bool rootTracked;
        public PoseSample root;
        public List<JointSample> joints = new List<JointSample>();
    }

    [Header("OSC Target")]
    [SerializeField] private string remoteHost = "127.0.0.1";
    [SerializeField] private int remotePort = 9000;
    [SerializeField] private string oscAddress = "/quest/upper_body";

    [Header("Pose Sources")]
    [SerializeField] private Transform rootTransform;
    [SerializeField] private Transform hmdFallback;
    [SerializeField] private JointBinding[] torsoAndArmJoints;

    [Header("Simulation")]
    [SerializeField, Tooltip("Generate a synthetic head pose when no XR device or fallback transform is available.")]
    private bool simulateHmdWhenUnavailable = true;
    [SerializeField, Tooltip("Base position offset for the simulated HMD pose in meters.")]
    private Vector3 simulatedHmdOffset = new Vector3(0f, 1.6f, 0f);
    [SerializeField, Tooltip("Oscillation amplitude for the simulated HMD pose in meters.")]
    private Vector3 simulatedHmdAmplitude = new Vector3(0.05f, 0.02f, 0.05f);
    [SerializeField, Tooltip("Oscillation amplitude for the simulated HMD yaw in degrees.")]
    private float simulatedYawAmplitude = 15f;

    [Header("Send Rate")]
    [SerializeField, Tooltip("How often to send packets (seconds)."), Min(0.001f)]
    private float sendInterval = 1f / 60f;

    private UdpClient udpClient;
    private double lastSendTime;

    private void Awake()
    {
        udpClient = new UdpClient();
        udpClient.EnableBroadcast = false;
        udpClient.Client.Blocking = false;
        udpClient.Connect(remoteHost, remotePort);
        lastSendTime = Time.realtimeSinceStartupAsDouble;
    }

    private void OnDestroy()
    {
        udpClient?.Dispose();
        udpClient = null;
    }

    private void Update()
    {
        if (udpClient == null)
        {
            return;
        }

        double now = Time.realtimeSinceStartupAsDouble;
        if (now - lastSendTime < sendInterval)
        {
            return;
        }

        if (!TryGetHmdPose(now, out Pose hmdPose))
        {
            return; // Without a valid HMD pose we skip the packet to keep downstream data clean.
        }

        PosePacket packet = BuildPacket(now, hmdPose);
        string json = JsonUtility.ToJson(packet);
        byte[] osc = BuildOscMessage(oscAddress, json);

        try
        {
            udpClient.Send(osc, osc.Length);
            lastSendTime = now;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"OSC send failed: {ex.Message}");
        }
    }

    private PosePacket BuildPacket(double timestamp, Pose hmdPose)
    {
        PosePacket packet = new PosePacket
        {
            timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
            hmd = new PoseSample
            {
                position = hmdPose.position,
                rotation = hmdPose.rotation
            }
        };

        if (rootTransform != null)
        {
            packet.rootTracked = rootTransform.gameObject.activeInHierarchy;
            packet.root = new PoseSample
            {
                position = rootTransform.position,
                rotation = rootTransform.rotation
            };
        }
        else
        {
            packet.rootTracked = false;
        }

        if (torsoAndArmJoints != null)
        {
            foreach (JointBinding binding in torsoAndArmJoints)
            {
                if (binding == null || string.IsNullOrEmpty(binding.jointName) || !binding.IsValid)
                {
                    continue;
                }

                packet.joints.Add(new JointSample
                {
                    name = binding.jointName,
                    pose = new PoseSample
                    {
                        position = binding.transform.position,
                        rotation = binding.transform.rotation
                    }
                });
            }
        }

        return packet;
    }

    private bool TryGetHmdPose(double sampleTime, out Pose pose)
    {
        if (TryGetDevicePose(XRNode.Head, out pose))
        {
            return true;
        }

        if (hmdFallback != null)
        {
            pose = new Pose(hmdFallback.position, hmdFallback.rotation);
            return true;
        }

        if (simulateHmdWhenUnavailable)
        {
            pose = GenerateSimulatedHmdPose(sampleTime);
            return true;
        }

        pose = Pose.identity;
        return false;
    }

    private Pose GenerateSimulatedHmdPose(double sampleTime)
    {
        float t = (float)sampleTime;
        Vector3 position = simulatedHmdOffset + new Vector3(
            Mathf.Sin(t) * simulatedHmdAmplitude.x,
            Mathf.Sin(t * 0.5f + Mathf.PI / 4f) * simulatedHmdAmplitude.y,
            Mathf.Cos(t) * simulatedHmdAmplitude.z
        );

        Quaternion rotation = Quaternion.Euler(
            0f,
            Mathf.Sin(t * 0.7f) * simulatedYawAmplitude,
            0f
        );

        return new Pose(position, rotation);
    }

    private static bool TryGetDevicePose(XRNode node, out Pose pose)
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(node);
        if (device.isValid &&
            device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 position) &&
            device.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rotation))
        {
            pose = new Pose(position, rotation);
            return true;
        }

        pose = Pose.identity;
        return false;
    }

    private static byte[] BuildOscMessage(string address, string stringPayload)
    {
        List<byte> buffer = new List<byte>(address.Length + stringPayload.Length + 16);
        AppendOscString(buffer, address);
        AppendOscString(buffer, ",s");
        AppendOscString(buffer, stringPayload);
        return buffer.ToArray();
    }

    private static void AppendOscString(List<byte> buffer, string value)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(value);
        buffer.AddRange(bytes);
        buffer.Add(0);
        while (buffer.Count % 4 != 0)
        {
            buffer.Add(0);
        }
    }
}
