import { HttpErrorResponse } from '@angular/common/http';
import { Component, Input, OnInit } from '@angular/core';
import { FormGroup, Validators,FormBuilder } from '@angular/forms';
import {Router, RouterModule} from '@angular/router';
import { Observable } from 'rxjs';
import { ServletService } from '../Service/servlet.service';
import { Result } from '../model/results.model';
import { NgxSpinnerService } from "ngx-spinner";

@Component({
  selector: 'app-search-ui',
  templateUrl: './search-ui.component.html',
  styleUrls: ['./search-ui.component.css']
})
export class SearchUIComponent implements OnInit {
  SearchForm!: FormGroup;
  title = 'Information Retrieval System';
  //fb: FormBuilder = new FormBuilder;
  myForm!: FormGroup;
  formData: any;
  countries = [];
  public query: string = "";
  user:any;
  results:Result[]=[];
  /*results: Result[] = [{
    "rank": 1, "relevanceScore": 2 ,
    "title": "First topic title",
    "imagePath": "https://www.google.com",
    "textFilePath": "https://www.google.com",
    "totRelDocs": 23,
    "rating":3
   },
 { 
 "rank": 2, 
 "relevanceScore": 2 ,
 "title": "Second topic title",
 "imagePath": "https://www.google.com",
 "textFilePath": "https://www.google.com",
 "totRelDocs": 24,
 "rating":4
},]*/
  
  constructor(private formBuilder: FormBuilder,private service : ServletService,
    private router: Router,private spinner: NgxSpinnerService,) { }
    
     
  
   _formValidate() {
    this.SearchForm = this.formBuilder.group({
      query: ['', Validators.required]});
  
   }
 
 
  ngOnInit() {
    this._formValidate();
    this.user=this.service.name;
    
  }
 logout(){
   console.log("Logging out,sending reset command,navigating to home page")
   this.router.navigate([ '/home' ]);
   //this.service.reset("reset").subscribe(data=>{});
   this.service.reset("reset").subscribe(
    (data) => {});

 }
  
  _formSubmit() {
    console.log("search submit");
  
   this.spinner.show();
   //console.log(typeof this.query + " "+this.query);
    if (this.query=="") {
      alert("Please enter query to proceed further")
      this.router.navigate(['/search']);
    } else {
      console.log("ïnside else")
      this.service.sendQuery(this.query)
      .subscribe(
          (data) => {
            this.results =[];
            console.log('Form submitted successfully'); 
            console.log(data); //change it to results variable of service class
            this.results=data;
            console.log(this.results.length);
            if (this.results.length==0) {
              alert("No Data found for given query.Please searchanother query")
              this.router.navigate(['/search']);
            } else {
              this.results =[];
              this.results=data;
              console.log("service call selse"+this.results)
              this.spinner.hide();
              
              this.service.results=this.results;
              this.service.getobj(this.results);
                 this.router.navigate([ '/results' ])
            }
           
           
          },
          (error: HttpErrorResponse) => {
            console.log("inside error")
              console.log(error);
             // this.router.navigate([ '/results' ])
          }
      );  
      
    }
    }
   
}
