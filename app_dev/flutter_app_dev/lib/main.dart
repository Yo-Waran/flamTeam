import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:arkit_plugin/arkit_plugin.dart';
import 'package:vector_math/vector_math_64.dart';
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const CupertinoApp(
      theme: CupertinoThemeData(
        primaryColor: CupertinoColors.systemBlue,
        brightness: Brightness.light,
      ),
      home: ARView(),
    );
  }
}

class ARView extends StatefulWidget {
  const ARView({super.key});

  @override
  State<ARView> createState() => _ARViewState();
}

class _ARViewState extends State<ARView> {
  late ARKitController arkitController;
  bool isPlacing = false;

  @override
  void dispose() {
    arkitController.dispose();
    super.dispose();
  }

  void _onARKitViewCreated(ARKitController arkitController) {
    this.arkitController = arkitController;
    this.arkitController.onAddNodeForAnchor = _handleAddAnchor;
  }

  void _handleAddAnchor(ARKitAnchor anchor) {
    if (anchor is ARKitPlaneAnchor) {
      _addPlane(arkitController, anchor);
    }
  }

  void _addPlane(ARKitController controller, ARKitPlaneAnchor anchor) {
    final material = ARKitMaterial(
      diffuse: ARKitMaterialProperty.color(CupertinoColors.systemBlue),
      transparency: 0.5,
    );

    final plane = ARKitPlane(
      width: anchor.extent.x,
      height: anchor.extent.z,
      materials: [material],
    );

    final node = ARKitNode(
      geometry: plane,
      position: Vector3(0, 0, 0),
    );

    controller.add(node, parentNodeName: anchor.nodeName);
  }

  void _placeCube() {
    if (!isPlacing) {
      setState(() {
        isPlacing = true;
      });

      final material = ARKitMaterial(
        diffuse: ARKitMaterialProperty.color(CupertinoColors.systemGreen),
        specular: ARKitMaterialProperty.color(CupertinoColors.white),
      );

      final cube = ARKitBox(
        width: 0.1,
        height: 0.1,
        length: 0.1,
        materials: [material],
      );

      final node = ARKitNode(
        geometry: cube,
        position: Vector3(0, 0, -0.5),
      );

      arkitController.add(node);
    }
  }

  @override
  Widget build(BuildContext context) {
    return CupertinoPageScaffold(
      navigationBar: const CupertinoNavigationBar(
        middle: Text('AR Cube'),
      ),
      child: Stack(
        children: [
          ARKitSceneView(
            onARKitViewCreated: _onARKitViewCreated,
            planeDetection: ARPlaneDetection.horizontal,
          ),
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: CupertinoButton(
                color: CupertinoColors.systemOrange,
                onPressed: _placeCube,
                child: const Text(
                  'Place Cube',
                  style: TextStyle(color: CupertinoColors.white),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}