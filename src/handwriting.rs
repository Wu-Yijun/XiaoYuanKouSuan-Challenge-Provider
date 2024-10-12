use core::f32;
use std::{collections::HashMap, thread::sleep, time::Duration};

use enigo::{Enigo, Mouse};

use crate::Config;

#[derive(Debug)]
pub struct HandWriting {
    pub supported: Vec<char>,
    pub character: HashMap<char, Vec<HandWritingShape>>,
}

#[derive(Debug)]
pub struct Writer {
    enigo: Enigo,
    x: f32,
    y: f32,
}
impl Default for Writer {
    fn default() -> Self {
        Self {
            enigo: Enigo::new(&enigo::Settings::default()).unwrap(),
            x: 0.0,
            y: 0.0,
        }
    }
}

impl Writer {
    pub fn write(&mut self, writing: &HandWriting, config: &Config, content: String) {
        println!("Write {content}");
        let x = (config.handwriting_pos.0 / config.display_scale) as i32 + 10;
        let y = (config.handwriting_pos.1 / config.display_scale) as i32 + 10;
        self.enigo.move_mouse(x, y, enigo::Coordinate::Abs).unwrap();
        self.write_string(writing, config, &content, config.font_size / config.display_scale);
    }
    pub fn write_string(&mut self, writing: &HandWriting, config: &Config, s: &str, size: f32) {
        for c in s.chars() {
            self.write_sized(writing, config, &c, size);
            self.move_to(0.4, 0.0, size);
        }
    }
    pub fn write_sized(&mut self, writing: &HandWriting, config: &Config, c: &char, size: f32) {
        if let Some(vhs) = writing.character.get(c) {
            self.x = 0.0;
            self.y = 0.0;
            for hs in vhs {
                // sleep(Duration::from_millis(1));
                match hs {
                    HandWritingShape::PenDown(true) => self
                        .enigo
                        .button(enigo::Button::Left, enigo::Direction::Press)
                        .unwrap(),
                    HandWritingShape::PenDown(false) => self
                        .enigo
                        .button(enigo::Button::Left, enigo::Direction::Release)
                        .unwrap(),
                    HandWritingShape::MoveTo((x, y)) => self.move_to(*x, *y, size),
                    HandWritingShape::LineTo((x, y)) => {
                        let s = ((self.x - x) * (self.x - x) + (self.y - y) * (self.y - y)).sqrt();
                        let ns = (s / 0.1).ceil().max(1.0) as i32;
                        let dx = (x - self.x) / ns as f32;
                        let dy = (y - self.y) / ns as f32;
                        let x0 = self.x;
                        let y0 = self.y;
                        for i in 1..=ns {
                            self.move_to(x0 + i as f32 * dx, y0 + i as f32 * dy, size);
                            sleep(Duration::from_millis(config.line_speed as u64));
                        }
                    }
                    HandWritingShape::Arc {
                        radius: (cx, cy, d),
                        range,
                    } => {
                        let th1 = range.0 / 180.0 * f32::consts::PI;
                        let th2 = range.1 / 180.0 * f32::consts::PI;
                        let nth = ((th2 - th1) / 0.25).abs().ceil().max(1.0) as i32;
                        let dth = (th2 - th1) / nth as f32;
                        // println!("Arc from {th1} to {th2} step {dth} total {nth}");
                        for i in 0..=nth {
                            let x = cx + d / 2.0 * (th1 + i as f32 * dth).cos();
                            let y = cy - d / 2.0 * (th1 + i as f32 * dth).sin();
                            self.move_to(x, y, size);
                            sleep(Duration::from_millis(config.arc_speed as u64));
                        }
                    }
                }
            }
            self.enigo
                .button(enigo::Button::Left, enigo::Direction::Release)
                .unwrap();
            self.move_to(0.0, 0.0, size);
        } else {
            println!("Cannot find key: {c}");
        }
    }
    fn move_to(&mut self, x: f32, y: f32, size: f32) {
        self.enigo
            .move_mouse(
                ((x - self.x) * size) as i32,
                ((y - self.y) * size) as i32,
                enigo::Coordinate::Rel,
            )
            .unwrap();
        sleep(Duration::from_millis(1));
        // println!("Move from {},{} to {x},{y}", self.x, self.y);
        self.x = x;
        self.y = y;
    }
}

#[derive(Debug, Clone)]
pub enum HandWritingShape {
    PenDown(bool),
    MoveTo((f32, f32)),
    LineTo((f32, f32)),
    Arc {
        /// x, y, d
        radius: (f32, f32, f32),
        // start deg, end deg
        range: (f32, f32),
    },
}

const NUM_0: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.343, 0.369)),
    HandWritingShape::PenDown(true),
    HandWritingShape::Arc {
        radius: (0.500, 0.623, 0.313),
        range: (180.0, 360.0),
    },
    HandWritingShape::Arc {
        radius: (0.500, 0.369, 0.313),
        range: (0.0, 180.0),
    },
];
const NUM_1: [HandWritingShape; 3] = [
    HandWritingShape::MoveTo((0.500, 0.223)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.500, 0.778)),
];
const NUM_2: [HandWritingShape; 5] = [
    HandWritingShape::MoveTo((0.329, 0.375)),
    HandWritingShape::PenDown(true),
    HandWritingShape::Arc {
        radius: (0.500, 0.369, 0.294),
        range: (180.0, -39.26),
    },
    HandWritingShape::LineTo((0.365, 0.747)),
    HandWritingShape::LineTo((0.669, 0.747)),
];
const NUM_3: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.365, 0.361)),
    HandWritingShape::PenDown(true),
    HandWritingShape::Arc {
        radius: (0.500, 0.361, 0.269),
        range: (180.0, -86.0),
    },
    HandWritingShape::Arc {
        radius: (0.508, 0.643, 0.286),
        range: (90.0, -180.0),
    },
];
const NUM_4: [HandWritingShape; 8] = [
    HandWritingShape::MoveTo((0.572, 0.267)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.335, 0.615)),
    HandWritingShape::LineTo((0.705, 0.615)),
    HandWritingShape::PenDown(false),
    HandWritingShape::MoveTo((0.569, 0.229)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.569, 0.775)),
];
const NUM_5: [HandWritingShape; 8] = [
    HandWritingShape::MoveTo((0.400, 0.229)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.367, 0.500)),
    HandWritingShape::Arc {
        radius: (0.500, 0.626, 0.365),
        range: (136.0, -164.0),
    },
    HandWritingShape::PenDown(false),
    HandWritingShape::MoveTo((0.398, 0.247)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.666, 0.247)),
];
const NUM_6: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.599, 0.243)),
    HandWritingShape::PenDown(true),
    HandWritingShape::Arc {
        radius: (0.555, 0.447, 0.413),
        range: (81.0, 173.0),
    },
    HandWritingShape::Arc {
        radius: (0.500, 0.626, 0.365),
        range: (180.0, 540.0),
    },
];
const NUM_7: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.319, 0.241)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.686, 0.241)),
    HandWritingShape::LineTo((0.416, 0.789)),
];
const NUM_8: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.500, 0.505)),
    HandWritingShape::PenDown(true),
    HandWritingShape::Arc {
        radius: (0.500, 0.367, 0.276),
        range: (-90.0, 270.0),
    },
    HandWritingShape::Arc {
        radius: (0.500, 0.665, 0.320),
        range: (90.0, -270.0),
    },
];
const NUM_9: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.500, 0.505)),
    HandWritingShape::PenDown(true),
    HandWritingShape::Arc {
        radius: (0.487, 0.375, 0.319),
        range: (-12.0, 348.0),
    },
    HandWritingShape::Arc {
        radius: (0.390, 0.485, 0.530),
        range: (333.0, 272.0),
    },
];
const SIGN_GREAT_THAN: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.342, 0.192)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.650, 0.500)),
    HandWritingShape::LineTo((0.342, 0.802)),
];
const SIGN_LESS_THAN: [HandWritingShape; 4] = [
    HandWritingShape::MoveTo((0.658, 0.192)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.350, 0.500)),
    HandWritingShape::LineTo((0.658, 0.802)),
];
const SIGN_EQUAL: [HandWritingShape; 7] = [
    HandWritingShape::MoveTo((0.310, 0.404)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.667, 0.404)),
    HandWritingShape::PenDown(false),
    HandWritingShape::MoveTo((0.310, 0.540)),
    HandWritingShape::PenDown(true),
    HandWritingShape::LineTo((0.667, 0.540)),
];
impl Default for HandWriting {
    fn default() -> Self {
        let mut character: HashMap<char, Vec<HandWritingShape>> = HashMap::new();
        character.insert('0', NUM_0.into());
        character.insert('1', NUM_1.into());
        character.insert('2', NUM_2.into());
        character.insert('3', NUM_3.into());
        character.insert('4', NUM_4.into());
        character.insert('5', NUM_5.into());
        character.insert('6', NUM_6.into());
        character.insert('7', NUM_7.into());
        character.insert('8', NUM_8.into());
        character.insert('9', NUM_9.into());
        character.insert('>', SIGN_GREAT_THAN.into());
        character.insert('<', SIGN_LESS_THAN.into());
        character.insert('=', SIGN_EQUAL.into());
        let supported = character.keys().map(|k| k.to_owned()).collect();
        Self {
            supported,
            character,
        }
    }
}

#[test]
fn test_input() {
    let h = HandWriting::default();
    let mut wt = Writer::default();
    sleep(Duration::from_millis(1000));
    // wt.write_string(&h, "000000", 100.0);
    // wt.write_string(&h, "111111", 100.0);
    // wt.write_string(&h, "222222", 100.0);
    // wt.move_to(-7.0, 1.0, 100.0);
    // wt.write_string(&h, "333333", 100.0);
    // wt.write_string(&h, "444444", 100.0);
    // wt.write_string(&h, "555555", 100.0);
    // wt.move_to(-7.0, 1.0, 100.0);
    // wt.write_string(&h, "666666", 100.0);
    // wt.write_string(&h, "777777", 100.0);
    // wt.write_string(&h, "888888", 100.0);
    // wt.move_to(-7.0, 1.0, 100.0);
    // wt.write_string(&h, "999999", 100.0);
    // wt.write_string(&h, "1239876543210", 100.0);
    // wt.write_sized(&h, &'0', 200.0);
    // wt.write_sized(&h, &'1', 200.0);
    // wt.write_sized(&h, &'2', 200.0);
    // wt.write_sized(&h, &'3', 200.0);
    // wt.write_sized(&h, &'4', 200.0);
    // wt.write_sized(&h, &'5', 200.0);
    // wt.write_sized(&h, &'6', 200.0);
    // wt.write_sized(&h, &'7', 200.0);
    // wt.write_sized(&h, &'8', 200.0);
    // wt.write_sized(&h, &'9', 200.0);
}

#[test]
fn test_position() {
    let mut enigo = Enigo::new(&enigo::Settings::default()).unwrap();
    enigo.move_mouse(1024 * 4 / 5, 1024 * 4 / 5, enigo::Coordinate::Abs);
}
