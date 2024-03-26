use std::{
    cell::{Ref, RefCell, RefMut},
    sync::{Arc, Mutex},
};

use cgmath::SquareMatrix;
use log::warn;

use crate::model::{InstanceId, Model};

pub struct NodeHandle(usize);

pub struct SceneNode {
    local_transform: cgmath::Matrix4<f32>,
    cached_transform: cgmath::Matrix4<f32>,
    transform_changed: bool,
    parent: Option<NodeHandle>,
    children: Vec<NodeHandle>,
    model: Option<Arc<Mutex<Model>>>,
    instance_id: Option<InstanceId>,
}

impl Default for SceneNode {
    fn default() -> Self {
        Self {
            local_transform: cgmath::Matrix4::identity(),
            cached_transform: cgmath::Matrix4::identity(),
            transform_changed: false,
            parent: Default::default(),
            children: Default::default(),
            model: Default::default(),
            instance_id: Default::default(),
        }
    }
}

impl SceneNode {
    pub fn get_local_transform(&self) -> cgmath::Matrix4<f32> {
        self.local_transform
    }

    pub fn update_local_transform(&mut self, transform: cgmath::Matrix4<f32>) {
        self.local_transform = transform;
        self.transform_changed = true;
        if self.parent.is_none() {
            self.cached_transform = self.local_transform;
        }
    }

    pub fn get_global_transform(&self) -> Option<cgmath::Matrix4<f32>> {
        if self.transform_changed {
            None
        } else {
            Some(self.cached_transform)
        }
    }

    pub fn set_model(&mut self, model: Arc<Mutex<Model>>) {
        if let Some(_old_model) = self.model.take() {
            panic!("Cannot handle removing an old model yet!");
        }
        {
            let mut guard = model.lock().expect("Poisoned Mutex");
            let id = guard.new_instance();
            guard.update_instance(&id, self.cached_transform);
            self.instance_id = Some(id);
        }
        self.model = Some(model);
    }

    fn refresh_transform(&mut self, parent_transform: cgmath::Matrix4<f32>) {
        self.transform_changed = false;
        self.cached_transform = parent_transform * self.local_transform;
        if let Some(model) = &self.model {
            if let Some(id) = &self.instance_id {
                model
                    .lock()
                    .expect("Poisoned Mutex")
                    .update_instance(id, self.cached_transform)
            } else {
                warn!("Have model but no id, cannot update!");
            }
        }
    }
}

#[derive(Default)]
pub struct SceneTree {
    nodes: Vec<RefCell<SceneNode>>,
}

impl SceneTree {
    pub fn get(&self, node_handle: &NodeHandle) -> Option<Ref<'_, SceneNode>> {
        self.nodes.get(node_handle.0).map(|n| n.borrow())
    }

    pub fn get_mut(&mut self, node_handle: &NodeHandle) -> Option<RefMut<'_, SceneNode>> {
        self.nodes.get(node_handle.0).map(|n| n.borrow_mut())
    }

    pub fn new_node(&mut self) -> NodeHandle {
        let node_index = self.nodes.len();
        let new_node = RefCell::new(SceneNode::default());
        self.nodes.push(new_node);
        NodeHandle(node_index)
    }

    pub fn update_transforms(&mut self) {
        let mut to_update: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if n.borrow().transform_changed {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        while !to_update.is_empty() {
            let index = match to_update.pop() {
                Some(i) => i,
                None => {
                    warn!("Tried to pop from an empty to_update queue!");
                    continue;
                }
            };
            let mut node = match self.nodes.get(index) {
                Some(n) => n.borrow_mut(),
                None => {
                    warn!("Invalid index to update: {}", index);
                    continue;
                }
            };
            if let Some(parent_index) = &node.parent {
                if index != parent_index.0 {
                    if let Some(parent) = self.nodes.get(parent_index.0) {
                        node.update_local_transform(parent.borrow().cached_transform);
                    } else {
                        warn!("Node has invalid parent index: {}", parent_index.0);
                    }
                } else {
                    warn!("Node {} is it's own parent!", index);
                }
            }
            for child_index in &node.children {
                to_update.push(child_index.0)
            }
        }
    }
}
