package de.ovgu.ir.controller;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.websocket.server.PathParam;

import org.apache.lucene.queryparser.classic.ParseException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


import de.ovgu.ir.model.QueryData;
import de.ovgu.ir.service.IRP02AppService;
import de.ovgu.ir.model.RerankData;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:4200")
public class IRP02ApplicationController {
	private IRP02AppService service;
	private List<QueryData> list = new ArrayList<>();
	@Autowired
	public IRP02ApplicationController(IRP02AppService service) {
		this.service = service;
	}
	@PostMapping("/getQuery")
	public ResponseEntity<Object> getQueryData(@RequestBody String query){
		
		 try { 
			 this.list = service.getSearchData(query); } 
		 catch (ParseException | IOException e) 
		 { e.printStackTrace(); }
		 
		return new ResponseEntity<>(this.list, HttpStatus.OK);
	}
	@PostMapping("/reRankData")
	public ResponseEntity<Object> addratingsData(@RequestBody List<RerankData> obj){
		
		 try { this.list = service.getreRerankData(obj); } 
		 catch (ParseException | IOException e) { e.printStackTrace(); }
		return new ResponseEntity<>(this.list, HttpStatus.OK);
	}
	@PostMapping("/logout")
	public ResponseEntity<String> logout(@RequestBody String reset){
		System.out.println("Executing logout...");
		service.logout();
		System.out.println(reset);
		return new ResponseEntity<>("Succesfully Logged out", HttpStatus.OK);
	}
}
